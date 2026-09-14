# -*- coding: utf-8 -*-
"""
Privileged CARLA reward metrics V5 - CLEAN

Gộp behavior production từ reward_metrics_carla_v3 + V4 wheel-aware offroad.

Privileged only:
- lateral_error_m
- heading_error_rad
- forward_progress_m
- wheel-aware offroad
- stuck
- collision

KHÔNG đưa các metric này vào PPO observation.

Python 3.7 compatible.
"""

from __future__ import print_function

import math
import weakref

from simulation.carla_connection_v5 import carla


CARLA_SCALE = 10.0

STUCK_SPEED_THRESHOLD_MPS = 0.05
STUCK_COMMAND_THRESHOLD_MPS = 0.20
STUCK_TIME_THRESHOLD_S = 2.0

OFFROAD_MIN_POINTS_OUT = 2
OFFROAD_CONFIRM_TICKS = 3
FALLBACK_BBOX_SHRINK = 0.85

RAW_WHEEL_SPAN_CM_DETECT_THRESHOLD = 10.0


def _wrap_pi(angle_rad):
    return (
        (float(angle_rad) + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def _dot_xy(ax, ay, bx, by):
    return (
        float(ax) * float(bx)
        + float(ay) * float(by)
    )


class CollisionTrackerV5(object):
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.sensor = None

        self.collided = False
        self.event_count = 0
        self.last_impulse = 0.0
        self.last_frame = None

        world = vehicle.get_world()

        blueprint = (
            world.get_blueprint_library()
            .find("sensor.other.collision")
        )

        self.sensor = world.spawn_actor(
            blueprint,
            carla.Transform(),
            attach_to=vehicle,
        )

        weak_self = weakref.ref(self)

        self.sensor.listen(
            lambda event: CollisionTrackerV5._on_collision(
                weak_self,
                event,
            )
        )

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()

        if self is None:
            return

        impulse = event.normal_impulse

        magnitude = math.sqrt(
            float(impulse.x) ** 2
            + float(impulse.y) ** 2
            + float(impulse.z) ** 2
        )

        self.collided = True
        self.event_count += 1
        self.last_impulse = float(magnitude)
        self.last_frame = int(event.frame)

    def reset(self):
        self.collided = False
        self.event_count = 0
        self.last_impulse = 0.0
        self.last_frame = None

    def consume_collision(self):
        collided = bool(self.collided)
        self.collided = False
        return collided

    def destroy(self):
        if self.sensor is None:
            return

        try:
            self.sensor.stop()
        except Exception:
            pass

        try:
            self.sensor.destroy()
        except Exception:
            pass

        self.sensor = None


class RoadRewardMetricsV5(object):
    """
    Wheel-aware V5 privileged road metrics.

    Offroad:
        >= 2/4 wheel points outside Driving lane
        for >= 3 consecutive control ticks
        -> offroad=True
    """

    def __init__(
        self,
        vehicle,
        carla_scale=CARLA_SCALE,
        stuck_speed_threshold_mps=STUCK_SPEED_THRESHOLD_MPS,
        stuck_command_threshold_mps=STUCK_COMMAND_THRESHOLD_MPS,
        stuck_time_threshold_s=STUCK_TIME_THRESHOLD_S,
        offroad_min_points_out=OFFROAD_MIN_POINTS_OUT,
        offroad_confirm_ticks=OFFROAD_CONFIRM_TICKS,
    ):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.map = self.world.get_map()

        self.carla_scale = float(carla_scale)

        self.stuck_speed_threshold_mps = float(
            stuck_speed_threshold_mps
        )
        self.stuck_command_threshold_mps = float(
            stuck_command_threshold_mps
        )
        self.stuck_time_threshold_s = float(
            stuck_time_threshold_s
        )

        self.offroad_min_points_out = int(
            offroad_min_points_out
        )
        self.offroad_confirm_ticks = int(
            offroad_confirm_ticks
        )

        if self.offroad_min_points_out < 1:
            raise ValueError(
                "offroad_min_points_out must be >= 1."
            )

        if self.offroad_confirm_ticks < 1:
            raise ValueError(
                "offroad_confirm_ticks must be >= 1."
            )

        self.prev_location = None
        self.stuck_time_s = 0.0

        self.wheel_out_count = 0
        self.wheel_out_flags = (
            False,
            False,
            False,
            False,
        )
        self.wheel_offroad_consecutive_ticks = 0
        self.wheel_point_source = "unknown"
        self.center_offroad = False

        self._last_log_state = None
        self._wheel_local_cache = None
        self._wheel_position_unit_scale = None

        self.reset()
        self._cache_wheel_local_points()

    def reset(self):
        location = self.vehicle.get_location()

        self.prev_location = carla.Location(
            x=float(location.x),
            y=float(location.y),
            z=float(location.z),
        )

        self.stuck_time_s = 0.0

        self.wheel_out_count = 0
        self.wheel_out_flags = (
            False,
            False,
            False,
            False,
        )
        self.wheel_offroad_consecutive_ticks = 0
        self.center_offroad = False
        self._last_log_state = None

    def _driving_waypoint_projected(self):
        return self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

    def _driving_waypoint_exact(self):
        return self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )

    def _lateral_error_real_m(self, waypoint):
        if waypoint is None:
            return 0.0

        vehicle_location = self.vehicle.get_location()
        waypoint_location = waypoint.transform.location
        right = waypoint.transform.get_right_vector()

        dx = float(
            vehicle_location.x - waypoint_location.x
        )
        dy = float(
            vehicle_location.y - waypoint_location.y
        )

        lateral_carla_m = _dot_xy(
            dx,
            dy,
            right.x,
            right.y,
        )

        return float(
            lateral_carla_m / self.carla_scale
        )

    def _heading_error_rad(self, waypoint):
        if waypoint is None:
            return 0.0

        vehicle_yaw = math.radians(
            float(
                self.vehicle
                .get_transform()
                .rotation
                .yaw
            )
        )

        road_yaw = math.radians(
            float(
                waypoint
                .transform
                .rotation
                .yaw
            )
        )

        return _wrap_pi(
            vehicle_yaw - road_yaw
        )

    def _forward_progress_real_m(self, waypoint):
        current = self.vehicle.get_location()

        if self.prev_location is None:
            self.prev_location = carla.Location(
                x=float(current.x),
                y=float(current.y),
                z=float(current.z),
            )
            return 0.0

        dx = float(
            current.x - self.prev_location.x
        )
        dy = float(
            current.y - self.prev_location.y
        )

        if waypoint is None:
            transform = self.vehicle.get_transform()
            forward = transform.get_forward_vector()
        else:
            forward = waypoint.transform.get_forward_vector()

        progress_carla_m = _dot_xy(
            dx,
            dy,
            forward.x,
            forward.y,
        )

        self.prev_location = carla.Location(
            x=float(current.x),
            y=float(current.y),
            z=float(current.z),
        )

        return float(
            progress_carla_m / self.carla_scale
        )

    def _update_stuck(
        self,
        speed_mps,
        target_speed_cmd_mps,
        dt,
    ):
        wants_to_move = (
            float(target_speed_cmd_mps)
            >= self.stuck_command_threshold_mps
        )

        almost_not_moving = (
            float(speed_mps)
            <= self.stuck_speed_threshold_mps
        )

        if wants_to_move and almost_not_moving:
            self.stuck_time_s += max(
                float(dt),
                0.0,
            )
        else:
            self.stuck_time_s = 0.0

        return bool(
            self.stuck_time_s
            >= self.stuck_time_threshold_s
        )

    # ------------------------------------------------------------------
    # Wheel-aware offroad
    # ------------------------------------------------------------------

    @staticmethod
    def _world_to_local_xy(
        transform,
        world_x,
        world_y,
        world_z,
    ):
        dx = (
            float(world_x)
            - float(transform.location.x)
        )
        dy = (
            float(world_y)
            - float(transform.location.y)
        )
        dz = (
            float(world_z)
            - float(transform.location.z)
        )

        yaw = math.radians(
            float(transform.rotation.yaw)
        )
        c = math.cos(yaw)
        s = math.sin(yaw)

        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        return (
            float(local_x),
            float(local_y),
            float(dz),
        )

    @staticmethod
    def _local_to_world_xy(
        transform,
        local_x,
        local_y,
        local_z=0.0,
    ):
        yaw = math.radians(
            float(transform.rotation.yaw)
        )
        c = math.cos(yaw)
        s = math.sin(yaw)

        return carla.Location(
            x=(
                float(transform.location.x)
                + c * float(local_x)
                - s * float(local_y)
            ),
            y=(
                float(transform.location.y)
                + s * float(local_x)
                + c * float(local_y)
            ),
            z=(
                float(transform.location.z)
                + float(local_z)
            ),
        )

    @staticmethod
    def _detect_raw_world_unit_scale(raw_points):
        xs = [float(p[0]) for p in raw_points]
        ys = [float(p[1]) for p in raw_points]

        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        max_span = max(
            abs(span_x),
            abs(span_y),
        )

        if (
            max_span
            > RAW_WHEEL_SPAN_CM_DETECT_THRESHOLD
        ):
            return (
                0.01,
                span_x,
                span_y,
                "cm_observed_dirty_build",
            )

        return (
            1.0,
            span_x,
            span_y,
            "meters_api_contract",
        )

    def _cache_wheel_local_points(self):
        try:
            transform = self.vehicle.get_transform()
            wheels = list(
                self.vehicle
                .get_physics_control()
                .wheels
            )

            raw_points = []

            for wheel in wheels[:4]:
                pos = getattr(
                    wheel,
                    "position",
                    None,
                )

                if pos is None:
                    raw_points = []
                    break

                raw_points.append(
                    (
                        float(pos.x),
                        float(pos.y),
                        float(pos.z),
                    )
                )

            if len(raw_points) == 4:
                (
                    unit_scale,
                    raw_span_x,
                    raw_span_y,
                    unit_mode,
                ) = self._detect_raw_world_unit_scale(
                    raw_points
                )

                local_points = []

                for raw_x, raw_y, raw_z in raw_points:
                    local_points.append(
                        self._world_to_local_xy(
                            transform,
                            raw_x * unit_scale,
                            raw_y * unit_scale,
                            raw_z * unit_scale,
                        )
                    )

                lx = [
                    p[0] for p in local_points
                ]
                ly = [
                    p[1] for p in local_points
                ]

                local_span_x = max(lx) - min(lx)
                local_span_y = max(ly) - min(ly)

                max_local_abs = max(
                    max(
                        abs(p[0]),
                        abs(p[1]),
                    )
                    for p in local_points
                )

                sane = (
                    0.5
                    <= max(
                        local_span_x,
                        local_span_y,
                    )
                    <= 6.0
                    and max_local_abs <= 5.0
                )

                if sane:
                    self._wheel_local_cache = tuple(
                        local_points
                    )
                    self._wheel_position_unit_scale = float(
                        unit_scale
                    )
                    self.wheel_point_source = (
                        "physics_world_normalized_cached"
                    )

                    print(
                        "V5 WHEEL GEOMETRY | source={} | "
                        "unit_mode={} | raw_span=({:.3f},{:.3f}) | "
                        "scale={} | local_span=({:.3f},{:.3f}) | "
                        "local_xy={}".format(
                            self.wheel_point_source,
                            unit_mode,
                            raw_span_x,
                            raw_span_y,
                            unit_scale,
                            local_span_x,
                            local_span_y,
                            [
                                (
                                    round(p[0], 4),
                                    round(p[1], 4),
                                )
                                for p in self._wheel_local_cache
                            ],
                        )
                    )
                    return

                print(
                    "V5 WHEEL GEOMETRY WARNING | "
                    "normalized geometry failed sanity check | "
                    "unit_mode={} | scale={} | "
                    "local_span=({:.3f},{:.3f}) | "
                    "max_local_abs={:.3f} | "
                    "using bbox fallback".format(
                        unit_mode,
                        unit_scale,
                        local_span_x,
                        local_span_y,
                        max_local_abs,
                    )
                )

        except Exception as error:
            print(
                "V5 WHEEL GEOMETRY WARNING | "
                "physics wheel position unavailable: {} | "
                "using bbox fallback".format(
                    error
                )
            )

        self._wheel_local_cache = None
        self._wheel_position_unit_scale = None
        self.wheel_point_source = "bbox_fallback"

    def _fallback_local_points(self):
        bbox = self.vehicle.bounding_box

        cx = float(bbox.location.x)
        cy = float(bbox.location.y)
        cz = float(bbox.location.z)

        ex = (
            abs(float(bbox.extent.x))
            * FALLBACK_BBOX_SHRINK
        )
        ey = (
            abs(float(bbox.extent.y))
            * FALLBACK_BBOX_SHRINK
        )

        return (
            (cx + ex, cy - ey, cz),
            (cx + ex, cy + ey, cz),
            (cx - ex, cy - ey, cz),
            (cx - ex, cy + ey, cz),
        )

    def _wheel_points(self):
        transform = self.vehicle.get_transform()

        if (
            self._wheel_local_cache is not None
            and len(self._wheel_local_cache) == 4
        ):
            local_points = self._wheel_local_cache
            self.wheel_point_source = (
                "physics_world_normalized_cached"
            )
        else:
            local_points = (
                self._fallback_local_points()
            )
            self.wheel_point_source = (
                "bbox_fallback"
            )

        return [
            self._local_to_world_xy(
                transform,
                local_x,
                local_y,
                local_z,
            )
            for (
                local_x,
                local_y,
                local_z,
            ) in local_points
        ]

    def _is_driving(self, location):
        return (
            self.map.get_waypoint(
                location,
                project_to_road=False,
                lane_type=carla.LaneType.Driving,
            )
            is not None
        )

    def _update_wheel_offroad(self):
        points = self._wheel_points()

        flags = tuple(
            not self._is_driving(point)
            for point in points
        )

        count = int(
            sum(
                1
                for flag in flags
                if flag
            )
        )

        if count >= self.offroad_min_points_out:
            self.wheel_offroad_consecutive_ticks += 1
        else:
            self.wheel_offroad_consecutive_ticks = 0

        confirmed = bool(
            self.wheel_offroad_consecutive_ticks
            >= self.offroad_confirm_ticks
        )

        self.wheel_out_flags = flags
        self.wheel_out_count = count

        state = (
            count,
            self.wheel_offroad_consecutive_ticks,
            confirmed,
        )

        if (
            state != self._last_log_state
            and (count > 0 or confirmed)
        ):
            print(
                "V5 WHEEL ROAD | out={}/4 | flags={} | "
                "confirm={}/{} | source={} | "
                "offroad={}".format(
                    count,
                    "".join(
                        "1" if x else "0"
                        for x in flags
                    ),
                    self.wheel_offroad_consecutive_ticks,
                    self.offroad_confirm_ticks,
                    self.wheel_point_source,
                    confirmed,
                )
            )

        self._last_log_state = state

        return confirmed

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        speed_mps,
        target_speed_cmd_mps,
        dt,
        collision=False,
    ):
        projected_wp = (
            self._driving_waypoint_projected()
        )
        exact_wp = (
            self._driving_waypoint_exact()
        )

        self.center_offroad = (
            exact_wp is None
        )

        lateral_error_m = (
            self._lateral_error_real_m(
                projected_wp
            )
        )

        heading_error_rad = (
            self._heading_error_rad(
                projected_wp
            )
        )

        forward_progress_m = (
            self._forward_progress_real_m(
                projected_wp
            )
        )

        stuck = self._update_stuck(
            speed_mps=speed_mps,
            target_speed_cmd_mps=(
                target_speed_cmd_mps
            ),
            dt=dt,
        )

        offroad = self._update_wheel_offroad()

        return {
            "lateral_error_m": float(
                lateral_error_m
            ),
            "heading_error_rad": float(
                heading_error_rad
            ),
            "forward_progress_m": float(
                forward_progress_m
            ),

            "collision": bool(collision),
            "offroad": bool(offroad),
            "center_offroad": bool(
                self.center_offroad
            ),

            "stuck": bool(stuck),
            "stuck_time_s": float(
                self.stuck_time_s
            ),

            "waypoint_valid": bool(
                projected_wp is not None
            ),

            "wheel_out_count": int(
                self.wheel_out_count
            ),
            "wheel_out_flags": tuple(
                self.wheel_out_flags
            ),
            "wheel_offroad_consecutive_ticks": int(
                self.wheel_offroad_consecutive_ticks
            ),
            "wheel_offroad_confirm_ticks": int(
                self.offroad_confirm_ticks
            ),
            "wheel_offroad_min_points_out": int(
                self.offroad_min_points_out
            ),
            "wheel_point_source": str(
                self.wheel_point_source
            ),
        }
