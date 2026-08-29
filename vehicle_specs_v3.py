# -*- coding: utf-8 -*-
"""
Canonical V3 vehicle/camera constants.

Only measured/frozen values live here.
Unknown dynamics/timing values remain None until GĐ4-GĐ7 calibration.
"""

import math
from dataclasses import dataclass
from typing import Optional


VEHICLE_BLUEPRINT_ID = "vehicle.ty.automav3"

CARLA_GEOMETRY_SCALE = 10.0
REAL_MAX_SPEED_MPS = 1.0
CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ
COMMAND_TX_HZ = 50.0
FEEDBACK_RX_HZ = 50.0
STATE_READ_HZ = 50.0
ENCODER_FEEDBACK_HZ = 50.0
SERVO_FEEDBACK_HZ = 50.0
IMU_FPS = 50.0


@dataclass(frozen=True)
class RealVehicleSpecs:
    wheelbase_m: float = 0.258
    wheel_diameter_m: float = 0.067
    mass_kg: float = 4.0

    track_front_m: float = 0.185
    track_rear_m: float = 0.182
    max_steer_angle_deg: float = 35.0
    max_speed_mps: float = REAL_MAX_SPEED_MPS

    # Latest measured real camera geometry.
    camera_from_rear_axle_m: float = 0.245
    camera_lateral_m: float = 0.0
    camera_height_m: float = 0.159
    camera_ground_distance_m: float = 0.292

    # Measured real IMU mount geometry.
    # REAL coordinate convention:
    #   origin X/Y = rear axle center
    #   +X forward, +Y left, +Z up
    # IMU is centered laterally, halfway along the wheelbase,
    # and 0.070 m above the ground.
    imu_from_rear_axle_m: float = wheelbase_m / 2.0
    imu_lateral_m: float = 0.0
    imu_height_from_ground_m: float = 0.070

    # Pending GĐ4-GĐ7 measurements.
    servo_center_to_full_time_s: Optional[float] = None
    normal_speed_mps: Optional[float] = None
    accel_0_to_normal_time_s: Optional[float] = None
    coast_normal_to_zero_time_s: Optional[float] = None
    control_latency_s: Optional[float] = None
    imu_gyro_z_abs_max_rad_s: Optional[float] = None
    imu_accel_x_abs_max_mps2: Optional[float] = None


REAL = RealVehicleSpecs()

CARLA_TARGET_WHEELBASE_M = (
    REAL.wheelbase_m * CARLA_GEOMETRY_SCALE
)
CARLA_TARGET_WHEEL_DIAMETER_M = (
    REAL.wheel_diameter_m * CARLA_GEOMETRY_SCALE
)
CARLA_TARGET_WHEEL_RADIUS_M = (
    CARLA_TARGET_WHEEL_DIAMETER_M / 2.0
)

# WheelPhysicsControl.radius is centimeters in CARLA 0.9.13.
CARLA_WHEEL_RADIUS_CM = (
    REAL.wheel_diameter_m * 0.5 * 100.0 * CARLA_GEOMETRY_SCALE
)
CARLA_FRONT_MAX_STEER_DEG = REAL.max_steer_angle_deg
CARLA_REAR_MAX_STEER_DEG = 0.0

# Measured model-local markers for vehicle.ty.automav3.
MODEL_REAR_AXLE_X = -1.408498
MODEL_GROUND_Z = -0.127419

FRONT_CAMERA_WIDTH = 160
FRONT_CAMERA_HEIGHT = 80
FRONT_CAMERA_FPS = 30.0
FRONT_CAMERA_FOV_DEG = 125.0

FRONT_CAMERA_X = (
    MODEL_REAR_AXLE_X
    + REAL.camera_from_rear_axle_m * CARLA_GEOMETRY_SCALE
)
FRONT_CAMERA_Y = (
    REAL.camera_lateral_m * CARLA_GEOMETRY_SCALE
)
FRONT_CAMERA_Z = (
    MODEL_GROUND_Z
    + REAL.camera_height_m * CARLA_GEOMETRY_SCALE
)
FRONT_CAMERA_PITCH_DEG = -math.degrees(
    math.atan2(
        REAL.camera_height_m,
        REAL.camera_ground_distance_m,
    )
)
FRONT_CAMERA_YAW_DEG = 0.0
FRONT_CAMERA_ROLL_DEG = 0.0


# IMU local transform for vehicle.ty.automav3.
# Convert measured REAL mount geometry to the 10x CARLA model,
# using the measured model-local rear-axle and ground markers.
IMU_X = (
    MODEL_REAR_AXLE_X
    + REAL.imu_from_rear_axle_m * CARLA_GEOMETRY_SCALE
)
IMU_Y = (
    REAL.imu_lateral_m * CARLA_GEOMETRY_SCALE
)
IMU_Z = (
    MODEL_GROUND_Z
    + REAL.imu_height_from_ground_m * CARLA_GEOMETRY_SCALE
)

IMU_PITCH_DEG = 0.0
IMU_YAW_DEG = 0.0
IMU_ROLL_DEG = 0.0


if __name__ == "__main__":
    print("Blueprint:", VEHICLE_BLUEPRINT_ID)
    print("Real max speed:", REAL.max_speed_mps, "m/s")
    print("Real max steer:", REAL.max_steer_angle_deg, "deg")
    print("CARLA wheel radius:", CARLA_WHEEL_RADIUS_CM, "cm")
    print(
        "Camera XYZ:",
        FRONT_CAMERA_X,
        FRONT_CAMERA_Y,
        FRONT_CAMERA_Z,
    )
    print("Camera pitch:", FRONT_CAMERA_PITCH_DEG, "deg")
    print(
        "Real IMU XYZ from rear-axle/ground:",
        REAL.imu_from_rear_axle_m,
        REAL.imu_lateral_m,
        REAL.imu_height_from_ground_m,
    )
    print("CARLA IMU XYZ:", IMU_X, IMU_Y, IMU_Z)
