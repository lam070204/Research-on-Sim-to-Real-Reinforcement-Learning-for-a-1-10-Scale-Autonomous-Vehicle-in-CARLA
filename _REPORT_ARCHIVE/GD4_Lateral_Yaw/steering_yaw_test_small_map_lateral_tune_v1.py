# -*- coding: utf-8 -*-
"""
GĐ4 - Test 4: Steering / Yaw baseline for CARLA (small-map)

Mục tiêu:
- So sánh REAL <-> SIM yaw-rate tại:
    speed_cmd = 0.30, 0.50, 0.70 m/s
    steer_cmd = +12 deg, -12 deg
- Chưa tune lateral trong file này.
- Không dùng ax để quyết định PASS/FAIL.
- Longitudinal chỉ dùng cấu hình PROVISIONAL V2B để giữ speed gần REAL:
    torque scale      = 0.80
    max brake torque  = 560
    autobox            = True
    gear_switch_time   = 0.0

Small-map strategy:
- MỖI (speed, steer) là một case độc lập.
- Reset về spawn trước mỗi case.
- Pre-speed thẳng: 2.5 s.
- Steering hold: 1.4 s.
- Phân tích 1.0 s cuối của steering hold.
=> tránh chạy quá xa trên map nhỏ.

REAL reference:
- Tự tìm Log/test4_steering_yaw_*_summary.csv và gộp các speed.
- Nếu không tìm thấy, dùng fallback đã đo ngày 2026-08-27.

Chạy:
    python .\dynamics_calibration\tests\steering_yaw_test_small_map.py --spawn 2
"""

from __future__ import print_function

import argparse
import csv
import glob
import math
import sys
from datetime import datetime
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from simulation.carla_connection_v3 import carla
from simulation.simulation_settings_v3 import HOST, PORT
from vehicle_specs_v3 import (
    VEHICLE_BLUEPRINT_ID,
    CARLA_GEOMETRY_SCALE,
    CARLA_WHEEL_RADIUS_CM,
    CARLA_FRONT_MAX_STEER_DEG,
    CARLA_REAR_MAX_STEER_DEG,
    REAL,
)
from action_controller_v3 import (
    ActionControllerV3,
    vehicle_planar_speed_real_mps,
)


# =============================================================================
# Test profile
# =============================================================================

CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

SPEED_COMMANDS_MPS = [0.30, 0.50, 0.70]
STEER_COMMANDS_DEG = [+12.0, -12.0]

# GĐ4 lateral Tune V1 - one-shot directional steering calibration.
#
# Derived from the 6-case baseline by minimizing the maximum relative
# yaw-magnitude error separately for + and - steering directions:
#   positive gain ~= 1.1245
#   negative gain ~= 1.4639
#
# These gains represent the REAL steering linkage / wheel-angle asymmetry
# that is not captured by symmetric CARLA steering geometry.
# They are PROVISIONAL until this single validation run passes.
STEER_GAIN_POSITIVE = 1.1245
STEER_GAIN_NEGATIVE = 1.4639

STOP_BEFORE_S = 1.0
PRE_SPEED_S = 2.5
STEER_HOLD_S = 1.4
ANALYSIS_WINDOW_S = 1.0

ANALYSIS_SAMPLES = int(round(ANALYSIS_WINDOW_S * CONTROL_HZ))

SETTLE_S = 0.5
SPAWN_Z_OFFSET_M = 0.25
EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"

# Safety only; not a calibration threshold.
MAX_DISTANCE_FROM_SPAWN_CARLA_M = 60.0
# Custom vehicle geometry is 10x; its actor origin can settle several CARLA meters
# below the raw spawn-point Z even while the wheels remain correctly on the road.
# 20 CARLA m ~= 2 real-equivalent m and still catches a genuine fall off the map.
MAX_DROP_FROM_SPAWN_Z_M = 20.0


# =============================================================================
# Provisional longitudinal configuration
# =============================================================================

TORQUE_SCALE_PROVISIONAL = 0.80
MAX_BRAKE_TORQUE_PROVISIONAL = 560.0
AUTOBOX_PROVISIONAL = True
GEAR_SWITCH_TIME_PROVISIONAL_S = 0.0


# =============================================================================
# REAL references from uploaded GĐ4 logs.
# Used only if Log/test4_..._summary.csv is unavailable.
# =============================================================================

REAL_FALLBACK = {
    (0.30, +12.0): {
        "speed_mean_mps": 0.29170588235294115,
        "steer_feedback_mean_deg": 11.558941176470588,
        "yaw_rate_mean_rad_s": 0.23117060784313725,
        "yaw_rate_std_rad_s": 0.025896408918673206,
        "samples": 51,
    },
    (0.30, -12.0): {
        "speed_mean_mps": 0.30911764705882355,
        "steer_feedback_mean_deg": -12.14735294117647,
        "yaw_rate_mean_rad_s": -0.2945028823529412,
        "yaw_rate_std_rad_s": 0.024999325517819124,
        "samples": 51,
    },
    (0.50, +12.0): {
        "speed_mean_mps": 0.49541176470588233,
        "steer_feedback_mean_deg": 11.833,
        "yaw_rate_mean_rad_s": 0.4313272941176471,
        "yaw_rate_std_rad_s": 0.03609063093784542,
        "samples": 51,
    },
    (0.50, -12.0): {
        "speed_mean_mps": 0.5161176470588236,
        "steer_feedback_mean_deg": -12.167,
        "yaw_rate_mean_rad_s": -0.5330954705882353,
        "yaw_rate_std_rad_s": 0.039183350734388674,
        "samples": 51,
    },
    (0.70, +12.0): {
        "speed_mean_mps": 0.6845882352941176,
        "steer_feedback_mean_deg": 11.833,
        "yaw_rate_mean_rad_s": 0.5967914117647058,
        "yaw_rate_std_rad_s": 0.057044992242326224,
        "samples": 51,
    },
    (0.70, -12.0): {
        "speed_mean_mps": 0.7316274509803922,
        "steer_feedback_mean_deg": -12.167,
        "yaw_rate_mean_rad_s": -0.7498071176470589,
        "yaw_rate_std_rad_s": 0.06956690331375892,
        "samples": 51,
    },
}


# =============================================================================
# Helpers
# =============================================================================

def mean(values):
    if not values:
        return float("nan")
    return float(sum(float(v) for v in values) / len(values))


def std_population(values):
    if not values:
        return float("nan")

    m = mean(values)

    return float(
        math.sqrt(
            sum((float(v) - m) ** 2 for v in values)
            / len(values)
        )
    )


def pct_error(sim, real):
    sim = float(sim)
    real = float(real)

    if not math.isfinite(real) or abs(real) < 1e-12:
        return float("nan")

    return float((sim - real) / abs(real) * 100.0)


def magnitude_pct_error(sim, real):
    sim = abs(float(sim))
    real = abs(float(real))

    if not math.isfinite(real) or real < 1e-12:
        return float("nan")

    return float((sim - real) / real * 100.0)


def ensure_output_dir():
    path = PROJECT_ROOT / "dynamics_calibration" / "sim_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_yaw_rate_rad_s(vehicle):
    """
    CARLA Vehicle.get_angular_velocity() returns degrees/s.
    Convert Z yaw rate to rad/s.
    """
    angular = vehicle.get_angular_velocity()
    return float(math.radians(float(angular.z)))


def get_longitudinal_accel_real_mps2(vehicle):
    """
    Diagnostic only. ax is NOT used for lateral PASS/FAIL.
    """
    acc = vehicle.get_acceleration()
    forward = vehicle.get_transform().get_forward_vector()

    long_acc_carla = (
        float(acc.x) * float(forward.x)
        + float(acc.y) * float(forward.y)
        + float(acc.z) * float(forward.z)
    )

    return float(long_acc_carla / float(CARLA_GEOMETRY_SCALE))


def bicycle_yaw_model_rad_s(speed_real_mps, steer_deg):
    """
    Kinematic bicycle reference using REAL-equivalent quantities.

    Since CARLA geometry and speed are both scaled by 10x,
    yaw rate is scale-invariant:
        (10*v) / (10*L) = v/L
    """
    wheelbase = float(REAL.wheelbase_m)

    if wheelbase <= 0.0:
        return float("nan")

    return float(
        float(speed_real_mps)
        / wheelbase
        * math.tan(math.radians(float(steer_deg)))
    )


def steer_deg_to_normalized(steer_deg):
    """
    Map REAL steering command degrees -> CARLA normalized steer.

    Tune V1 uses two directional gains because the REAL yaw response is
    consistently stronger for negative steering (~24-27% across all speeds),
    while CARLA baseline is almost symmetric.

    IMPORTANT:
    - This is a steering-linkage calibration, not a change to tire/COM.
    - ±12 deg REAL command becomes an effective CARLA wheel command:
        +12 deg -> +13.494 deg
        -12 deg -> -17.567 deg
    """
    max_deg = float(CARLA_FRONT_MAX_STEER_DEG)

    if max_deg <= 0.0:
        raise ValueError("CARLA_FRONT_MAX_STEER_DEG phải > 0.")

    steer_deg = float(steer_deg)

    if steer_deg > 0.0:
        effective_deg = steer_deg * float(STEER_GAIN_POSITIVE)
    elif steer_deg < 0.0:
        effective_deg = steer_deg * float(STEER_GAIN_NEGATIVE)
    else:
        effective_deg = 0.0

    value = effective_deg / max_deg
    return max(-1.0, min(+1.0, value))


def effective_carla_steer_deg(real_steer_cmd_deg):
    real_steer_cmd_deg = float(real_steer_cmd_deg)

    if real_steer_cmd_deg > 0.0:
        return real_steer_cmd_deg * float(STEER_GAIN_POSITIVE)
    if real_steer_cmd_deg < 0.0:
        return real_steer_cmd_deg * float(STEER_GAIN_NEGATIVE)
    return 0.0


# =============================================================================
# Vehicle setup
# =============================================================================

def apply_vehicle_geometry(vehicle):
    if vehicle.type_id != VEHICLE_BLUEPRINT_ID:
        raise RuntimeError(
            "Sai vehicle blueprint: {} != {}".format(
                vehicle.type_id,
                VEHICLE_BLUEPRINT_ID,
            )
        )

    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)

    if len(wheels) < 4:
        raise RuntimeError(
            "Vehicle chỉ có {} wheels.".format(len(wheels))
        )

    for wheel in wheels[:4]:
        wheel.radius = float(CARLA_WHEEL_RADIUS_CM)

    wheels[0].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
    wheels[1].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
    wheels[2].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)
    wheels[3].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)

    physics.wheels = wheels
    vehicle.apply_physics_control(physics)


def apply_provisional_longitudinal(vehicle):
    """
    GĐ4 longitudinal is NOT frozen yet because REAL ax is pending.
    This only preserves the current provisional speed behavior while
    Test 4 isolates lateral/yaw behavior.
    """
    physics = vehicle.get_physics_control()

    physics.torque_curve = [
        carla.Vector2D(
            float(point.x),
            float(point.y) * TORQUE_SCALE_PROVISIONAL,
        )
        for point in list(physics.torque_curve)
    ]

    wheels = list(physics.wheels)

    for wheel in wheels:
        wheel.max_brake_torque = float(
            MAX_BRAKE_TORQUE_PROVISIONAL
        )

    physics.wheels = wheels
    physics.use_gear_autobox = bool(AUTOBOX_PROVISIONAL)
    physics.gear_switch_time = float(
        GEAR_SWITCH_TIME_PROVISIONAL_S
    )

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
        "autobox": bool(applied.use_gear_autobox),
        "gear_switch_time_s": float(applied.gear_switch_time),
    }


def stop_vehicle(vehicle):
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
            reverse=False,
            manual_gear_shift=False,
        )
    )

    vehicle.set_target_velocity(
        carla.Vector3D(x=0.0, y=0.0, z=0.0)
    )

    vehicle.set_target_angular_velocity(
        carla.Vector3D(x=0.0, y=0.0, z=0.0)
    )


def reset_vehicle_to_spawn(
    world,
    vehicle,
    spawn_transform,
):
    stop_vehicle(vehicle)

    vehicle.set_transform(
        carla.Transform(
            carla.Location(
                x=float(spawn_transform.location.x),
                y=float(spawn_transform.location.y),
                z=float(spawn_transform.location.z)
                + SPAWN_Z_OFFSET_M,
            ),
            spawn_transform.rotation,
        )
    )

    stop_vehicle(vehicle)

    for _ in range(
        max(1, int(round(SETTLE_S * CONTROL_HZ)))
    ):
        world.tick()

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )
    )

    world.tick()


def spawn_vehicle(world, spawn_number):
    spawn_points = list(
        world.get_map().get_spawn_points()
    )

    if not spawn_points:
        raise RuntimeError("Map không có spawn points.")

    idx = int(spawn_number) - 1

    if idx < 0 or idx >= len(spawn_points):
        raise ValueError(
            "--spawn {} không hợp lệ; map có {} spawn.".format(
                spawn_number,
                len(spawn_points),
            )
        )

    matches = world.get_blueprint_library().filter(
        VEHICLE_BLUEPRINT_ID
    )

    if not matches:
        raise RuntimeError(
            "Không tìm thấy blueprint {}".format(
                VEHICLE_BLUEPRINT_ID
            )
        )

    original = spawn_points[idx]

    transform = carla.Transform(
        carla.Location(
            x=float(original.location.x),
            y=float(original.location.y),
            z=float(original.location.z)
            + SPAWN_Z_OFFSET_M,
        ),
        original.rotation,
    )

    vehicle = world.try_spawn_actor(
        matches[0],
        transform,
    )

    if vehicle is None:
        raise RuntimeError(
            "Không spawn được vehicle tại spawn {}.".format(
                spawn_number
            )
        )

    world.tick()

    apply_vehicle_geometry(vehicle)
    tune_info = apply_provisional_longitudinal(vehicle)

    world.tick()

    verify = vehicle.get_physics_control()

    print(
        "    LONGITUDINAL PROVISIONAL | torque={} | brake={} | "
        "autobox={} | gear_switch={:.3f}s".format(
            tune_info["torque_curve"],
            tune_info["max_brake_torque"],
            bool(verify.use_gear_autobox),
            float(verify.gear_switch_time),
        )
    )

    return vehicle, original


# =============================================================================
# Logging
# =============================================================================

def safety_check(vehicle, spawn_transform):
    location = vehicle.get_location()

    dx = float(location.x) - float(
        spawn_transform.location.x
    )
    dy = float(location.y) - float(
        spawn_transform.location.y
    )

    distance = math.sqrt(dx * dx + dy * dy)

    if distance > MAX_DISTANCE_FROM_SPAWN_CARLA_M:
        raise RuntimeError(
            "Vehicle đi quá xa spawn ({:.1f} m CARLA). "
            "Dừng để tránh map contamination.".format(
                distance
            )
        )

    if (
        float(location.z)
        < float(spawn_transform.location.z)
        - MAX_DROP_FROM_SPAWN_Z_M
    ):
        raise RuntimeError(
            "Vehicle bị rơi khỏi map: z={:.3f}, spawn_z={:.3f}, drop={:.3f} CARLA m.".format(
                float(location.z),
                float(spawn_transform.location.z),
                float(spawn_transform.location.z) - float(location.z),
            )
        )


def sample_row(
    case_name,
    phase,
    phase_elapsed_s,
    total_elapsed_s,
    speed_cmd_mps,
    steer_cmd_deg,
    steer_cmd_norm,
    vehicle,
    info,
    sim_frame,
):
    control = vehicle.get_control()
    speed = vehicle_planar_speed_real_mps(vehicle)
    yaw_rate = get_yaw_rate_rad_s(vehicle)

    return {
        "case": case_name,
        "phase": phase,
        "jetson_elapsed_s": float(phase_elapsed_s),
        "time_ms": int(round(total_elapsed_s * 1000.0)),
        "speed_cmd_mps": float(speed_cmd_mps),
        "speed_mps": float(speed),

        # REAL-schema-compatible lateral fields
        "steer_cmd_deg": float(steer_cmd_deg),
        "steer_fb_deg": "",
        "gyro_z_rad_s": float(yaw_rate),
        "linear_accel_x_m_s2": float(
            get_longitudinal_accel_real_mps2(vehicle)
        ),
        "yaw_model_rad_s": float(
            bicycle_yaw_model_rad_s(
                speed,
                steer_cmd_deg,
            )
        ),

        # CARLA diagnostics
        "sim_frame": int(sim_frame),
        "effective_carla_steer_cmd_deg": float(
            effective_carla_steer_deg(steer_cmd_deg)
        ),
        "requested_steer_norm": float(
            steer_cmd_norm
        ),
        "applied_steer_norm": float(
            info["applied_steer_cmd"]
        ),
        "applied_steer_angle_deg": float(
            info["applied_steer_angle_deg"]
        ),
        "requested_speed_cmd_mps": float(
            info["requested_speed_cmd_mps"]
        ),
        "applied_speed_cmd_mps": float(
            info["applied_speed_cmd_mps"]
        ),
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "carla_steer": float(control.steer),
        "gear": int(control.gear),
        "manual_gear_shift": int(
            bool(control.manual_gear_shift)
        ),
    }


def run_phase(
    world,
    vehicle,
    controller,
    spawn_transform,
    rows,
    case_name,
    phase,
    speed_cmd_mps,
    steer_cmd_deg,
    duration_s,
    total_elapsed_s,
):
    steer_norm = steer_deg_to_normalized(
        steer_cmd_deg
    )

    n_ticks = max(
        1,
        int(round(float(duration_s) * CONTROL_HZ))
    )

    phase_rows = []

    for i in range(n_ticks):
        info = controller.step(
            steer_cmd=float(steer_norm),
            speed_cmd_mps=float(speed_cmd_mps),
            dt=DT,
        )

        frame = int(world.tick())
        safety_check(vehicle, spawn_transform)

        phase_elapsed_s = (i + 1) * DT
        total_elapsed_s += DT

        row = sample_row(
            case_name=case_name,
            phase=phase,
            phase_elapsed_s=phase_elapsed_s,
            total_elapsed_s=total_elapsed_s,
            speed_cmd_mps=speed_cmd_mps,
            steer_cmd_deg=steer_cmd_deg,
            steer_cmd_norm=steer_norm,
            vehicle=vehicle,
            info=info,
            sim_frame=frame,
        )

        rows.append(row)
        phase_rows.append(row)

    return total_elapsed_s, phase_rows


# =============================================================================
# REAL reference loading
# =============================================================================

def load_real_reference():
    """
    Merge all available REAL Test 4 summary files.
    Newer duplicate (speed, steer) rows override older ones.
    """
    pattern = str(
        PROJECT_ROOT
        / "Log"
        / "test4_steering_yaw_*_summary.csv"
    )

    paths = sorted(glob.glob(pattern))
    refs = {}

    for path in paths:
        with open(
            path,
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    key = (
                        round(float(row["speed_cmd_mps"]), 2),
                        round(float(row["steer_cmd_deg"]), 1),
                    )

                    refs[key] = {
                        "speed_mean_mps": float(
                            row["speed_mean_mps"]
                        ),
                        "steer_feedback_mean_deg": float(
                            row["steer_feedback_mean_deg"]
                        ),
                        "yaw_rate_mean_rad_s": float(
                            row["yaw_rate_mean_rad_s"]
                        ),
                        "yaw_rate_std_rad_s": float(
                            row["yaw_rate_std_rad_s"]
                        ),
                        "samples": int(float(row["samples"])),
                        "source": str(path),
                    }
                except Exception:
                    continue

    if not refs:
        refs = {}

        for key, value in REAL_FALLBACK.items():
            refs[key] = dict(value)
            refs[key]["source"] = "fallback_20260827"

    return refs, paths


# =============================================================================
# Summary / comparison
# =============================================================================

def summarize_case(
    speed_cmd,
    steer_deg,
    steering_rows,
):
    if not steering_rows:
        raise RuntimeError(
            "Không có steering rows cho case."
        )

    window = steering_rows[-ANALYSIS_SAMPLES:]

    speeds = [
        float(row["speed_mps"])
        for row in window
    ]

    yaws = [
        float(row["gyro_z_rad_s"])
        for row in window
    ]

    model_yaws = [
        float(row["yaw_model_rad_s"])
        for row in window
    ]

    applied_angles = [
        float(row["applied_steer_angle_deg"])
        for row in window
    ]

    throttles = [
        float(row["throttle"])
        for row in window
    ]

    brakes = [
        float(row["brake"])
        for row in window
    ]

    return {
        "speed_cmd_mps": float(speed_cmd),
        "steer_cmd_deg": float(steer_deg),
        "speed_mean_mps": mean(speeds),
        "speed_std_mps": std_population(speeds),
        "applied_steer_angle_mean_deg": mean(
            applied_angles
        ),
        "yaw_rate_mean_rad_s": mean(yaws),
        "yaw_rate_std_rad_s": std_population(yaws),
        "yaw_rate_abs_mean_rad_s": abs(mean(yaws)),
        "yaw_model_mean_rad_s": mean(model_yaws),
        "throttle_mean": mean(throttles),
        "brake_mean": mean(brakes),
        "samples": int(len(window)),
    }


def build_compare(sim_summary, real_refs):
    rows = []

    for sim in sim_summary:
        key = (
            round(float(sim["speed_cmd_mps"]), 2),
            round(float(sim["steer_cmd_deg"]), 1),
        )

        real = real_refs.get(key)

        if real is None:
            continue

        real_yaw = float(
            real["yaw_rate_mean_rad_s"]
        )
        sim_yaw = float(
            sim["yaw_rate_mean_rad_s"]
        )

        real_speed = float(
            real["speed_mean_mps"]
        )
        sim_speed = float(
            sim["speed_mean_mps"]
        )

        rows.append({
            "speed_cmd_mps": float(
                sim["speed_cmd_mps"]
            ),
            "steer_cmd_deg": float(
                sim["steer_cmd_deg"]
            ),

            "real_speed_mean_mps": real_speed,
            "sim_speed_mean_mps": sim_speed,
            "speed_error_mps": sim_speed - real_speed,
            "speed_error_pct": pct_error(
                sim_speed,
                real_speed,
            ),

            "real_steer_feedback_mean_deg": float(
                real["steer_feedback_mean_deg"]
            ),
            "sim_applied_steer_angle_mean_deg": float(
                sim["applied_steer_angle_mean_deg"]
            ),

            "real_yaw_rate_mean_rad_s": real_yaw,
            "sim_yaw_rate_mean_rad_s": sim_yaw,
            "yaw_signed_error_rad_s": sim_yaw - real_yaw,
            "yaw_magnitude_error_pct": magnitude_pct_error(
                sim_yaw,
                real_yaw,
            ),

            "real_yaw_rate_std_rad_s": float(
                real["yaw_rate_std_rad_s"]
            ),
            "sim_yaw_rate_std_rad_s": float(
                sim["yaw_rate_std_rad_s"]
            ),

            "sim_yaw_model_mean_rad_s": float(
                sim["yaw_model_mean_rad_s"]
            ),

            "real_samples": int(real["samples"]),
            "sim_samples": int(sim["samples"]),
            "real_source": str(real["source"]),
        })

    return rows


def build_asymmetry(compare_rows):
    output = []

    for speed in SPEED_COMMANDS_MPS:
        pos = None
        neg = None

        for row in compare_rows:
            if abs(
                float(row["speed_cmd_mps"]) - speed
            ) > 1e-9:
                continue

            if float(row["steer_cmd_deg"]) > 0:
                pos = row
            elif float(row["steer_cmd_deg"]) < 0:
                neg = row

        if pos is None or neg is None:
            continue

        real_pos = abs(
            float(pos["real_yaw_rate_mean_rad_s"])
        )
        real_neg = abs(
            float(neg["real_yaw_rate_mean_rad_s"])
        )

        sim_pos = abs(
            float(pos["sim_yaw_rate_mean_rad_s"])
        )
        sim_neg = abs(
            float(neg["sim_yaw_rate_mean_rad_s"])
        )

        output.append({
            "speed_cmd_mps": float(speed),
            "real_abs_yaw_pos_rad_s": real_pos,
            "real_abs_yaw_neg_rad_s": real_neg,
            "real_neg_over_pos_ratio": (
                real_neg / real_pos
                if real_pos > 1e-12
                else float("nan")
            ),
            "sim_abs_yaw_pos_rad_s": sim_pos,
            "sim_abs_yaw_neg_rad_s": sim_neg,
            "sim_neg_over_pos_ratio": (
                sim_neg / sim_pos
                if sim_pos > 1e-12
                else float("nan")
            ),
        })

    return output


def write_csv(path, rows):
    if not rows:
        return

    with open(
        str(path),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_result(
    sim_summary,
    compare_rows,
    asymmetry_rows,
):
    print("")
    print("=" * 126)
    print("GĐ4 TEST 4 - STEERING / YAW RESULT")
    print("=" * 126)
    print(
        "{:<7} {:>7} | {:>9} {:>9} {:>8} | "
        "{:>10} {:>10} {:>10} | {:>9}".format(
            "speed",
            "steer",
            "REAL v",
            "SIM v",
            "v err%",
            "REAL yaw",
            "SIM yaw",
            "yaw err%",
            "SIM std",
        )
    )

    for row in compare_rows:
        print(
            "{:<7.2f} {:+7.1f} | {:>9.4f} {:>9.4f} {:+7.2f}% | "
            "{:+10.4f} {:+10.4f} {:+9.2f}% | {:>9.4f}".format(
                float(row["speed_cmd_mps"]),
                float(row["steer_cmd_deg"]),
                float(row["real_speed_mean_mps"]),
                float(row["sim_speed_mean_mps"]),
                float(row["speed_error_pct"]),
                float(row["real_yaw_rate_mean_rad_s"]),
                float(row["sim_yaw_rate_mean_rad_s"]),
                float(row["yaw_magnitude_error_pct"]),
                float(row["sim_yaw_rate_std_rad_s"]),
            )
        )

    print("=" * 126)

    if asymmetry_rows:
        print("")
        print("LEFT/RIGHT MAGNITUDE ASYMMETRY")
        print("-" * 86)
        print(
            "{:<7} | {:>13} {:>13} | {:>13} {:>13}".format(
                "speed",
                "REAL |-|/|+|",
                "SIM |-|/|+|",
                "REAL delta%",
                "SIM delta%",
            )
        )

        for row in asymmetry_rows:
            real_ratio = float(
                row["real_neg_over_pos_ratio"]
            )
            sim_ratio = float(
                row["sim_neg_over_pos_ratio"]
            )

            print(
                "{:<7.2f} | {:>13.3f} {:>13.3f} | "
                "{:+12.1f}% {:+12.1f}%".format(
                    float(row["speed_cmd_mps"]),
                    real_ratio,
                    sim_ratio,
                    (real_ratio - 1.0) * 100.0,
                    (sim_ratio - 1.0) * 100.0,
                )
            )

        print("-" * 86)

    print("")
    print(
        "NOTE: ax chỉ được ghi diagnostic; KHÔNG dùng ax để đánh giá Test 4."
    )
    print(
        "NOTE: đây là BASELINE lateral/yaw; chưa tune tire/COM/steering."
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spawn",
        type=int,
        default=2,
        help="Spawn point 1-based. Default: 2",
    )

    args = parser.parse_args()

    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)

    world = client.get_world()

    if EXPECTED_MAP_FRAGMENT not in str(
        world.get_map().name
    ):
        raise RuntimeError(
            "Sai map: {}. Cần map chứa '{}'.".format(
                world.get_map().name,
                EXPECTED_MAP_FRAGMENT,
            )
        )

    original_settings = world.get_settings()

    vehicle = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False

        world.apply_settings(settings)

        print("=" * 96)
        print(
            "GĐ4 - TEST 4 STEERING / YAW - SMALL MAP - LATERAL TUNE V1"
        )
        print("=" * 96)
        print("MAP       :", world.get_map().name)
        print("VEHICLE   :", VEHICLE_BLUEPRINT_ID)
        print(
            "CONTROL   : {:.1f} Hz | dt={:.3f} s".format(
                CONTROL_HZ,
                DT,
            )
        )
        print(
            "SCALE     : {:.1f}x".format(
                CARLA_GEOMETRY_SCALE
            )
        )
        print("SPAWN     :", args.spawn)
        print(
            "SPEEDS    : {}".format(
                SPEED_COMMANDS_MPS
            )
        )
        print(
            "STEERS    : {} deg".format(
                STEER_COMMANDS_DEG
            )
        )
        print(
            "PROFILE   : stop {:.1f}s | pre-speed {:.1f}s | "
            "steer {:.1f}s | analyze last {:.1f}s".format(
                STOP_BEFORE_S,
                PRE_SPEED_S,
                STEER_HOLD_S,
                ANALYSIS_WINDOW_S,
            )
        )
        print(
            "STEER TUNE: gain(+)= {:.4f} | gain(-)= {:.4f}".format(
                STEER_GAIN_POSITIVE,
                STEER_GAIN_NEGATIVE,
            )
        )
        print(
            "STEER MAP : REAL +12 -> CARLA {:+.3f} deg (norm {:+.6f}) | "
            "REAL -12 -> CARLA {:+.3f} deg (norm {:+.6f})".format(
                effective_carla_steer_deg(+12.0),
                steer_deg_to_normalized(+12.0),
                effective_carla_steer_deg(-12.0),
                steer_deg_to_normalized(-12.0),
            )
        )
        print("=" * 96)

        vehicle, spawn_transform = spawn_vehicle(
            world,
            args.spawn,
        )

        controller = ActionControllerV3(vehicle)

        real_refs, real_paths = load_real_reference()

        if real_paths:
            print("REAL REF FILES:")
            for path in real_paths:
                print("   ", path)
        else:
            print(
                "REAL REF   : fallback embedded from "
                "2026-08-27 logs"
            )

        rows = []
        sim_summary = []
        total_elapsed_s = 0.0

        for speed_cmd in SPEED_COMMANDS_MPS:
            for steer_deg in STEER_COMMANDS_DEG:
                case_name = "v{:.2f}_steer{:+.1f}".format(
                    speed_cmd,
                    steer_deg,
                )

                print("")
                print(
                    ">>> TEST speed={:.2f} m/s | steer={:+.1f} deg".format(
                        speed_cmd,
                        steer_deg,
                    )
                )

                reset_vehicle_to_spawn(
                    world,
                    vehicle,
                    spawn_transform,
                )

                controller.reset()
                world.tick()

                total_elapsed_s, _ = run_phase(
                    world=world,
                    vehicle=vehicle,
                    controller=controller,
                    spawn_transform=spawn_transform,
                    rows=rows,
                    case_name=case_name,
                    phase="stop",
                    speed_cmd_mps=0.0,
                    steer_cmd_deg=0.0,
                    duration_s=STOP_BEFORE_S,
                    total_elapsed_s=total_elapsed_s,
                )

                total_elapsed_s, pre_rows = run_phase(
                    world=world,
                    vehicle=vehicle,
                    controller=controller,
                    spawn_transform=spawn_transform,
                    rows=rows,
                    case_name=case_name,
                    phase="pre_speed",
                    speed_cmd_mps=speed_cmd,
                    steer_cmd_deg=0.0,
                    duration_s=PRE_SPEED_S,
                    total_elapsed_s=total_elapsed_s,
                )

                pre_window = pre_rows[-ANALYSIS_SAMPLES:]

                pre_speed_mean = mean(
                    [
                        float(r["speed_mps"])
                        for r in pre_window
                    ]
                )

                pre_speed_std = std_population(
                    [
                        float(r["speed_mps"])
                        for r in pre_window
                    ]
                )

                print(
                    "    PRE SPEED | {:.4f} ± {:.4f} m/s".format(
                        pre_speed_mean,
                        pre_speed_std,
                    )
                )

                total_elapsed_s, steer_rows = run_phase(
                    world=world,
                    vehicle=vehicle,
                    controller=controller,
                    spawn_transform=spawn_transform,
                    rows=rows,
                    case_name=case_name,
                    phase="steer_hold",
                    speed_cmd_mps=speed_cmd,
                    steer_cmd_deg=steer_deg,
                    duration_s=STEER_HOLD_S,
                    total_elapsed_s=total_elapsed_s,
                )

                summary = summarize_case(
                    speed_cmd=speed_cmd,
                    steer_deg=steer_deg,
                    steering_rows=steer_rows,
                )

                sim_summary.append(summary)

                print(
                    "    SIM YAW   | mean={:+.4f} rad/s | "
                    "std={:.4f} | speed={:.4f}".format(
                        float(
                            summary[
                                "yaw_rate_mean_rad_s"
                            ]
                        ),
                        float(
                            summary[
                                "yaw_rate_std_rad_s"
                            ]
                        ),
                        float(
                            summary["speed_mean_mps"]
                        ),
                    )
                )

                # Brief stop before teleport/reset.
                stop_vehicle(vehicle)
                world.tick()

        compare_rows = build_compare(
            sim_summary,
            real_refs,
        )

        asymmetry_rows = build_asymmetry(
            compare_rows
        )

        print_result(
            sim_summary=sim_summary,
            compare_rows=compare_rows,
            asymmetry_rows=asymmetry_rows,
        )

        output_dir = ensure_output_dir()
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        raw_path = output_dir / (
            "test4_steering_yaw_carla_{}.csv".format(
                stamp
            )
        )

        summary_path = output_dir / (
            "test4_steering_yaw_carla_{}_summary.csv".format(
                stamp
            )
        )

        compare_path = output_dir / (
            "test4_steering_yaw_real_vs_carla_{}.csv".format(
                stamp
            )
        )

        asymmetry_path = output_dir / (
            "test4_steering_yaw_asymmetry_{}.csv".format(
                stamp
            )
        )

        write_csv(raw_path, rows)
        write_csv(summary_path, sim_summary)
        write_csv(compare_path, compare_rows)
        write_csv(asymmetry_path, asymmetry_rows)

        print("")
        print("RAW CSV     :", raw_path)
        print("SUMMARY CSV :", summary_path)
        print("COMPARE CSV :", compare_path)
        print("ASYMM CSV   :", asymmetry_path)
        print("")
        print(
            "HOÀN TẤT TEST 4 LATERAL TUNE V1 VALIDATION. "
            "Gửi toàn bộ terminal output để PASS hoặc giữ lateral provisional."
        )

    finally:
        if vehicle is not None:
            try:
                stop_vehicle(vehicle)
                world.tick()
            except Exception:
                pass

            try:
                vehicle.destroy()
            except Exception:
                pass

        try:
            world.apply_settings(
                original_settings
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
