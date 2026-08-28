# -*- coding: utf-8 -*-
"""
GĐ4 longitudinal calibration - Tune V1.

CALIBRATION ONLY.
Chưa đưa các giá trị này vào PPO environment cho tới khi Test 1+2+3 xác nhận.

Baseline đo được:
- Acceleration SIM nhanh hơn REAL:
    0->0.30 : 0.96 s vs 1.72 s
    0->0.50 : 1.10 s vs 1.78 s
    0->0.70 : 1.28 s vs 1.60 s
- Full-stop SIM chậm hơn REAL:
    0.50->0 : 0.92 s vs 0.72 s
    0.70->0 : 1.52 s vs 1.24 s

Tune V1:
- Giữ mass = 1500 kg.
- Scale engine torque xuống 0.80.
  Chọn theo case 0->0.70, nơi steady error nhỏ nhất:
      1.28 / 1.60 = 0.80
- Tăng wheel max_brake_torque 450 -> 560.
  Chọn gần trung bình hai tỉ lệ full-stop:
      450 * mean(0.92/0.72, 1.52/1.24) ~= 563
  làm tròn 560.
- Giữ ActionController gains nguyên vẹn trong V1.

Mục tiêu:
Tách physics acceleration và physical braking trước.
Sau V1 mới chỉnh low-speed feed-forward / brake Kp nếu cần.
"""

from simulation.carla_connection_v3 import carla


ENGINE_TORQUE_SCALE_V1 = 0.80
MAX_BRAKE_TORQUE_V1 = 560.0


def apply_longitudinal_tune_v1(vehicle):
    physics = vehicle.get_physics_control()

    # Engine torque curve: preserve RPM locations, scale torque only.
    original_curve = list(physics.torque_curve)
    tuned_curve = []

    for point in original_curve:
        tuned_curve.append(
            carla.Vector2D(
                float(point.x),
                float(point.y) * ENGINE_TORQUE_SCALE_V1,
            )
        )

    physics.torque_curve = tuned_curve

    # Physical wheel braking.
    wheels = list(physics.wheels)
    for wheel in wheels:
        wheel.max_brake_torque = float(MAX_BRAKE_TORQUE_V1)

    physics.wheels = wheels
    vehicle.apply_physics_control(physics)

    # Read back for verification.
    applied = vehicle.get_physics_control()
    applied_curve = list(applied.torque_curve)
    applied_wheels = list(applied.wheels)

    return {
        "mass_kg": float(applied.mass),
        "torque_curve": [
            (float(p.x), float(p.y))
            for p in applied_curve
        ],
        "max_brake_torque": [
            float(w.max_brake_torque)
            for w in applied_wheels
        ],
    }
