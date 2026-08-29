# -*- coding: utf-8 -*-
"""
GĐ12.2 - Privileged CARLA reward metrics V3

Các biến này CHỈ dùng để tính reward / termination trong simulator:
    lateral_error_m
    heading_error_rad
    forward_progress_m
    offroad
    stuck
    collision

KHÔNG đưa chúng vào PPO observation.

Scale convention:
    CARLA length = real length * 10
    => mọi khoảng cách CARLA được chia CARLA_SCALE=10
    => heading/yaw không scale
"""

import math
import weakref

from simulation.carla_connection_v3 import carla


CARLA_SCALE = 10.0

STUCK_SPEED_THRESHOLD_MPS = 0.05
STUCK_COMMAND_THRESHOLD_MPS = 0.20
STUCK_TIME_THRESHOLD_S = 2.0


def _wrap_pi(angle_rad):
    return (
        (float(angle_rad) + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def _dot_xy(ax, ay, bx, by):
    return float(ax) * float(bx) + float(ay) * float(by)


class CollisionTrackerV3:
    """
    Lightweight privileged collision tracker for reward/termination.
    """

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
            lambda event: CollisionTrackerV3._on_collision(
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
        """
        Returns current collision flag, then clears only the flag.
        Event count remains available for diagnostics.
        """
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


class RoadRewardMetricsV3:
    """
    Stateful CARLA privileged metrics.

    Use pattern:
        metrics.reset(vehicle)
        ...
        m = metrics.update(
            speed_mps=...,
            target_speed_cmd_mps=...,
            dt=1/30,
        )
    """

    def __init__(
        self,
        vehicle,
        carla_scale=CARLA_SCALE,
        stuck_speed_threshold_mps=STUCK_SPEED_THRESHOLD_MPS,
        stuck_command_threshold_mps=STUCK_COMMAND_THRESHOLD_MPS,
        stuck_time_threshold_s=STUCK_TIME_THRESHOLD_S,
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

        self.prev_location = None
        self.stuck_time_s = 0.0

        self.reset()

    def reset(self):
        location = self.vehicle.get_location()

        self.prev_location = carla.Location(
            x=float(location.x),
            y=float(location.y),
            z=float(location.z),
        )

        self.stuck_time_s = 0.0

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
        """
        Signed lateral error relative to nearest driving waypoint center.

        Positive/negative sign follows waypoint right-vector direction.
        """
        if waypoint is None:
            return 0.0

        vehicle_location = self.vehicle.get_location()
        waypoint_location = waypoint.transform.location
        right = waypoint.transform.get_right_vector()

        dx = float(vehicle_location.x - waypoint_location.x)
        dy = float(vehicle_location.y - waypoint_location.y)

        lateral_carla_m = _dot_xy(
            dx,
            dy,
            right.x,
            right.y,
        )

        return float(
            lateral_carla_m
            / self.carla_scale
        )

    def _heading_error_rad(self, waypoint):
        """
        Vehicle yaw - road yaw, wrapped into [-pi, pi].
        """
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
        """
        Signed displacement since previous update projected onto road forward.

        Positive: moving along waypoint direction.
        Negative: moving backward.

        Returned in REAL-EQUIVALENT meters.
        """
        current = self.vehicle.get_location()

        if self.prev_location is None:
            self.prev_location = carla.Location(
                x=float(current.x),
                y=float(current.y),
                z=float(current.z),
            )
            return 0.0

        dx = float(current.x - self.prev_location.x)
        dy = float(current.y - self.prev_location.y)

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
            progress_carla_m
            / self.carla_scale
        )

    def _update_stuck(
        self,
        speed_mps,
        target_speed_cmd_mps,
        dt,
    ):
        """
        Stuck only counts when the agent is commanding meaningful motion.
        """
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

    def update(
        self,
        speed_mps,
        target_speed_cmd_mps,
        dt,
        collision=False,
    ):
        projected_wp = self._driving_waypoint_projected()
        exact_wp = self._driving_waypoint_exact()

        # Exact driving waypoint absent => outside Driving lane.
        offroad = exact_wp is None

        lateral_error_m = self._lateral_error_real_m(
            projected_wp
        )

        heading_error_rad = self._heading_error_rad(
            projected_wp
        )

        forward_progress_m = self._forward_progress_real_m(
            projected_wp
        )

        stuck = self._update_stuck(
            speed_mps=speed_mps,
            target_speed_cmd_mps=target_speed_cmd_mps,
            dt=dt,
        )

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
            "offroad": bool(offroad),
            "stuck": bool(stuck),
            "stuck_time_s": float(
                self.stuck_time_s
            ),
            "collision": bool(collision),
            "waypoint_valid": bool(
                projected_wp is not None
            ),
        }
