# -*- coding: utf-8 -*-
from __future__ import print_function

import math

from simulation.carla_connection_v3 import carla
from reward_metrics_carla_v3 import (
    RoadRewardMetricsV3,
    CARLA_SCALE,
    STUCK_SPEED_THRESHOLD_MPS,
    STUCK_COMMAND_THRESHOLD_MPS,
    STUCK_TIME_THRESHOLD_S,
)

OFFROAD_MIN_POINTS_OUT = 2
OFFROAD_CONFIRM_TICKS = 3
FALLBACK_BBOX_SHRINK = 0.85

# A normal passenger vehicle wheel span is nowhere near 10 m.
# Your CARLA 0.9.13-dirty build exposes wheel world positions whose
# inter-wheel deltas are ~260.5 and ~181.9, i.e. centimeters.
RAW_WHEEL_SPAN_CM_DETECT_THRESHOLD = 10.0


class RoadRewardMetricsV4(RoadRewardMetricsV3):
    """
    V4-only privileged wheel-aware offroad metric.

    Handles both CARLA variants:
      - wheel.position returned in world meters (official API contract)
      - wheel.position returned in world centimeters (observed 0.9.13-dirty)

    Unit detection is based on INTER-WHEEL SPAN, not absolute world location:
      span > 10 -> assume centimeters -> multiply by 0.01
      otherwise -> assume meters

    The normalized world wheel positions are converted to vehicle-local
    geometry once per spawned vehicle and cached.

    Rule:
      >= 2/4 wheel points outside Driving lane
      for 3 consecutive control ticks
      -> offroad=True

    Wheel positions remain privileged SIM-only data and are never added
    to the PPO observation.
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
        self.offroad_min_points_out = int(offroad_min_points_out)
        self.offroad_confirm_ticks = int(offroad_confirm_ticks)

        if self.offroad_min_points_out < 1:
            raise ValueError("offroad_min_points_out must be >= 1.")
        if self.offroad_confirm_ticks < 1:
            raise ValueError("offroad_confirm_ticks must be >= 1.")

        self.wheel_out_count = 0
        self.wheel_out_flags = (False, False, False, False)
        self.wheel_offroad_consecutive_ticks = 0
        self.wheel_point_source = "unknown"
        self.center_offroad_v3 = False
        self._last_log_state = None

        self._wheel_local_cache = None
        self._wheel_position_unit_scale = None

        super(RoadRewardMetricsV4, self).__init__(
            vehicle=vehicle,
            carla_scale=carla_scale,
            stuck_speed_threshold_mps=stuck_speed_threshold_mps,
            stuck_command_threshold_mps=stuck_command_threshold_mps,
            stuck_time_threshold_s=stuck_time_threshold_s,
        )

        self._cache_wheel_local_points()

    def reset(self):
        super(RoadRewardMetricsV4, self).reset()

        self.wheel_out_count = 0
        self.wheel_out_flags = (False, False, False, False)
        self.wheel_offroad_consecutive_ticks = 0
        self.center_offroad_v3 = False
        self._last_log_state = None

    @staticmethod
    def _world_to_local_xy(transform, world_x, world_y, world_z):
        dx = float(world_x) - float(transform.location.x)
        dy = float(world_y) - float(transform.location.y)
        dz = float(world_z) - float(transform.location.z)

        yaw = math.radians(float(transform.rotation.yaw))
        c = math.cos(yaw)
        s = math.sin(yaw)

        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        return (float(local_x), float(local_y), float(dz))

    @staticmethod
    def _local_to_world_xy(transform, local_x, local_y, local_z=0.0):
        yaw = math.radians(float(transform.rotation.yaw))
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
            z=float(transform.location.z) + float(local_z),
        )

    @staticmethod
    def _detect_raw_world_unit_scale(raw_points):
        """
        Detect units from wheel-to-wheel spacing.

        This deliberately ignores absolute map coordinates, because a vehicle
        can legitimately spawn hundreds of meters from world origin.
        """
        xs = [float(p[0]) for p in raw_points]
        ys = [float(p[1]) for p in raw_points]

        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        max_span = max(abs(span_x), abs(span_y))

        if max_span > RAW_WHEEL_SPAN_CM_DETECT_THRESHOLD:
            return 0.01, span_x, span_y, "cm_observed_dirty_build"

        return 1.0, span_x, span_y, "meters_api_contract"

    def _cache_wheel_local_points(self):
        try:
            transform = self.vehicle.get_transform()
            wheels = list(
                self.vehicle.get_physics_control().wheels
            )

            raw_points = []

            for wheel in wheels[:4]:
                pos = getattr(wheel, "position", None)
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
                ) = self._detect_raw_world_unit_scale(raw_points)

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

                # Geometry sanity check after normalization.
                lx = [p[0] for p in local_points]
                ly = [p[1] for p in local_points]

                local_span_x = max(lx) - min(lx)
                local_span_y = max(ly) - min(ly)
                max_local_abs = max(
                    max(abs(p[0]), abs(p[1]))
                    for p in local_points
                )

                sane = (
                    0.5 <= max(local_span_x, local_span_y) <= 6.0
                    and max_local_abs <= 5.0
                )

                if sane:
                    self._wheel_local_cache = tuple(local_points)
                    self._wheel_position_unit_scale = float(unit_scale)
                    self.wheel_point_source = (
                        "physics_world_normalized_cached"
                    )

                    print(
                        "V4 WHEEL GEOMETRY | source={} | "
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
                    "V4 WHEEL GEOMETRY WARNING | normalized wheel geometry "
                    "failed sanity check | unit_mode={} | scale={} | "
                    "local_span=({:.3f},{:.3f}) | max_local_abs={:.3f} | "
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
                "V4 WHEEL GEOMETRY WARNING | physics wheel position "
                "unavailable: {} | using bbox fallback".format(error)
            )

        self._wheel_local_cache = None
        self._wheel_position_unit_scale = None
        self.wheel_point_source = "bbox_fallback"

    def _fallback_local_points(self):
        bbox = self.vehicle.bounding_box

        cx = float(bbox.location.x)
        cy = float(bbox.location.y)
        cz = float(bbox.location.z)

        ex = abs(float(bbox.extent.x)) * FALLBACK_BBOX_SHRINK
        ey = abs(float(bbox.extent.y)) * FALLBACK_BBOX_SHRINK

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
            local_points = self._fallback_local_points()
            self.wheel_point_source = "bbox_fallback"

        return [
            self._local_to_world_xy(
                transform,
                local_x,
                local_y,
                local_z,
            )
            for local_x, local_y, local_z in local_points
        ]

    def _is_driving(self, location):
        return self.map.get_waypoint(
            location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        ) is not None

    def _update_wheel_offroad(self):
        points = self._wheel_points()

        flags = tuple(
            not self._is_driving(point)
            for point in points
        )
        count = int(sum(1 for flag in flags if flag))

        if count >= self.offroad_min_points_out:
            self.wheel_offroad_consecutive_ticks += 1
        else:
            self.wheel_offroad_consecutive_ticks = 0

        confirmed = (
            self.wheel_offroad_consecutive_ticks
            >= self.offroad_confirm_ticks
        )

        self.wheel_out_flags = flags
        self.wheel_out_count = count

        state = (
            count,
            self.wheel_offroad_consecutive_ticks,
            bool(confirmed),
        )

        if state != self._last_log_state and (count > 0 or confirmed):
            print(
                "V4 WHEEL ROAD | out={}/4 | flags={} | "
                "confirm={}/{} | source={} | offroad={}".format(
                    count,
                    "".join("1" if x else "0" for x in flags),
                    self.wheel_offroad_consecutive_ticks,
                    self.offroad_confirm_ticks,
                    self.wheel_point_source,
                    bool(confirmed),
                )
            )

        self._last_log_state = state
        return bool(confirmed)

    def update(
        self,
        speed_mps,
        target_speed_cmd_mps,
        dt,
        collision=False,
    ):
        metrics = super(RoadRewardMetricsV4, self).update(
            speed_mps=speed_mps,
            target_speed_cmd_mps=target_speed_cmd_mps,
            dt=dt,
            collision=collision,
        )

        self.center_offroad_v3 = bool(metrics["offroad"])

        metrics["offroad"] = self._update_wheel_offroad()

        metrics["wheel_out_count"] = int(self.wheel_out_count)
        metrics["wheel_out_flags"] = tuple(self.wheel_out_flags)
        metrics["wheel_offroad_consecutive_ticks"] = int(
            self.wheel_offroad_consecutive_ticks
        )
        metrics["wheel_offroad_confirm_ticks"] = int(
            self.offroad_confirm_ticks
        )
        metrics["wheel_offroad_min_points_out"] = int(
            self.offroad_min_points_out
        )
        metrics["wheel_point_source"] = str(self.wheel_point_source)
        metrics["center_offroad_v3"] = bool(self.center_offroad_v3)

        return metrics
