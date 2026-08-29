# -*- coding: utf-8 -*-
"""
GĐ4 longitudinal calibration - Tune V2.

Mục tiêu V2:
- GIỮ NGUYÊN physics Tune V1:
    torque scale = 0.80
    max_brake_torque = 560
- Loại bỏ drivetrain/autobox không giống xe thật:
    use_gear_autobox = False
    gear_switch_time = 0.0
    force manual gear = 1 ở mọi control tick

Lý do:
Diagnostic V1 cho thấy fresh vehicle giữ gear=0 khoảng 1.5 s dù throttle~1.0,
sau đó mới vào gear=1. Xe thật không có độ trễ "vào số" kiểu này.

V2 cố tình CHƯA đổi:
- torque thêm
- brake torque thêm
- FF/Kp/Ki
- damping
để đo riêng ảnh hưởng của drivetrain fix.
"""

from action_controller_v3 import ActionControllerV3
from simulation.carla_connection_v3 import carla


ENGINE_TORQUE_SCALE_V2 = 0.80
MAX_BRAKE_TORQUE_V2 = 560.0
FORCED_GEAR_V2 = 1


def _force_manual_gear_1(vehicle):
    control = vehicle.get_control()
    control.manual_gear_shift = True
    control.gear = int(FORCED_GEAR_V2)
    control.reverse = False
    vehicle.apply_control(control)


def apply_longitudinal_tune_v2(vehicle):
    physics = vehicle.get_physics_control()

    # Keep Tune V1 torque scaling.
    original_curve = list(physics.torque_curve)
    physics.torque_curve = [
        carla.Vector2D(
            float(point.x),
            float(point.y) * ENGINE_TORQUE_SCALE_V2,
        )
        for point in original_curve
    ]

    # Keep Tune V1 wheel braking.
    wheels = list(physics.wheels)
    for wheel in wheels:
        wheel.max_brake_torque = float(MAX_BRAKE_TORQUE_V2)
    physics.wheels = wheels

    # Remove the non-physical CARLA autobox delay.
    physics.use_gear_autobox = False
    physics.gear_switch_time = 0.0

    vehicle.apply_physics_control(physics)
    _force_manual_gear_1(vehicle)

    applied = vehicle.get_physics_control()
    control = vehicle.get_control()

    return {
        "mass_kg": float(applied.mass),
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
        "manual_gear_shift": bool(control.manual_gear_shift),
        "gear": int(control.gear),
    }


class ActionControllerV3TuneV2(ActionControllerV3):
    """
    Same speed controller as the project baseline,
    but force CARLA into fixed forward gear 1.

    The real RC drivetrain has no CARLA-style automatic gearbox.
    """

    def reset(self):
        super().reset()
        _force_manual_gear_1(self.vehicle)

    def step(self, steer_cmd, speed_cmd_mps, dt):
        info = super().step(
            steer_cmd=steer_cmd,
            speed_cmd_mps=speed_cmd_mps,
            dt=dt,
        )

        # super().step() applies a VehicleControl with default gear fields.
        # Re-apply the SAME throttle/brake/steer but lock gear before world.tick().
        _force_manual_gear_1(self.vehicle)

        info["forced_manual_gear"] = int(FORCED_GEAR_V2)
        return info
