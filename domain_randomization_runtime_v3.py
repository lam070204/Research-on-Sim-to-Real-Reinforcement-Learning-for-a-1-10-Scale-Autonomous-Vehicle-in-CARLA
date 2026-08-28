# -*- coding: utf-8 -*-
# GĐ6 runtime integration helpers.
# Python 3.7 compatible.


def _clip(value, low, high):
    return float(max(float(low), min(float(high), float(value))))


def apply_domain_physics_v3(vehicle, params, carla):
    physics = vehicle.get_physics_control()

    torque_scale = float(params["torque_scale"])
    physics.torque_curve = [
        carla.Vector2D(
            float(point.x),
            float(point.y) * torque_scale,
        )
        for point in list(physics.torque_curve)
    ]

    wheels = list(physics.wheels)
    brake_torque = float(params["max_brake_torque"])

    for wheel in wheels:
        wheel.max_brake_torque = brake_torque

    physics.wheels = wheels
    physics.use_gear_autobox = bool(params["use_gear_autobox"])
    physics.gear_switch_time = float(params["gear_switch_time_s"])

    vehicle.apply_physics_control(physics)

    applied = vehicle.get_physics_control()

    return {
        "torque_curve": [
            (float(p.x), float(p.y))
            for p in list(applied.torque_curve)
        ],
        "max_brake_torque": [
            float(w.max_brake_torque)
            for w in list(applied.wheels)
        ],
        "use_gear_autobox": bool(applied.use_gear_autobox),
        "gear_switch_time_s": float(applied.gear_switch_time),
    }


class DomainRandomizedActionControllerV3(object):
    def __init__(
        self,
        base_controller,
        steer_gain_positive,
        steer_gain_negative,
        extra_command_delay_ticks=0,
    ):
        self.base_controller = base_controller
        self.steer_gain_positive = float(steer_gain_positive)
        self.steer_gain_negative = float(steer_gain_negative)

        self.extra_command_delay_ticks = int(extra_command_delay_ticks)
        if self.extra_command_delay_ticks not in (0, 1):
            raise ValueError(
                "GĐ6 hiện chỉ hỗ trợ extra_command_delay_ticks = 0 hoặc 1."
            )

        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0
        self._delay_queue = []

    def _reset_delay_queue(self):
        self._delay_queue = [
            (0.0, 0.0)
            for _ in range(self.extra_command_delay_ticks)
        ]

    def reset(self):
        self.base_controller.reset()
        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0
        self._reset_delay_queue()

    def _physical_steer(self, logical_steer):
        logical_steer = _clip(logical_steer, -1.0, 1.0)

        if logical_steer > 0.0:
            physical = logical_steer * self.steer_gain_positive
        elif logical_steer < 0.0:
            physical = logical_steer * self.steer_gain_negative
        else:
            physical = 0.0

        return _clip(physical, -1.0, 1.0)

    def _apply_delay(self, steer_cmd, speed_cmd_mps):
        incoming = (
            _clip(steer_cmd, -1.0, 1.0),
            max(0.0, float(speed_cmd_mps)),
        )

        if self.extra_command_delay_ticks == 0:
            return incoming

        self._delay_queue.append(incoming)
        return self._delay_queue.pop(0)

    def step(self, steer_cmd, speed_cmd_mps, dt):
        policy_requested_steer = _clip(steer_cmd, -1.0, 1.0)
        policy_requested_speed = max(0.0, float(speed_cmd_mps))

        logical_steer, logical_speed = self._apply_delay(
            policy_requested_steer,
            policy_requested_speed,
        )

        physical_steer = self._physical_steer(logical_steer)

        base_info = self.base_controller.step(
            steer_cmd=physical_steer,
            speed_cmd_mps=logical_speed,
            dt=dt,
        )

        # Keep observation/reward command semantics in REAL logical space.
        self.prev_steer_cmd = float(logical_steer)
        self.prev_speed_cmd_mps = float(logical_speed)

        info = dict(base_info)
        info["requested_steer_cmd"] = float(policy_requested_steer)
        info["requested_speed_cmd_mps"] = float(policy_requested_speed)
        info["actual_steer_cmd"] = float(logical_steer)
        info["actual_speed_cmd_mps"] = float(logical_speed)

        info["gd6_physical_steer_cmd"] = float(physical_steer)
        info["gd6_carla_actual_steer_cmd"] = float(
            base_info.get("actual_steer_cmd", physical_steer)
        )
        info["gd6_extra_command_delay_ticks"] = int(
            self.extra_command_delay_ticks
        )
        info["gd6_steer_gain_positive"] = float(
            self.steer_gain_positive
        )
        info["gd6_steer_gain_negative"] = float(
            self.steer_gain_negative
        )

        return info
