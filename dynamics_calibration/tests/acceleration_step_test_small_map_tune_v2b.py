# -*- coding: utf-8 -*-
"""
GĐ4 - Test 2: CARLA Acceleration Step

Mục tiêu:
- Small-map adaptation of the REAL acceleration step:
    stop_before: 3 s
    accel step:  2.5 s
    stop_after:  0.5 s
- Same 0->target command, same 50 Hz timing; only the unnecessary tail is shortened
- target speed: 0.3 / 0.5 / 0.7 m/s
- steer_cmd = 0
- synchronous 50 Hz, dt = 0.02 s
- Dùng ActionControllerV3 hiện tại
- Không PPO, không VAE, không camera
- Xuất raw CSV + summary CSV + REAL-vs-SIM compare CSV

Metric được tính giống summary REAL:
- steady speed = 0.5 s cuối của đoạn acceleration 2.5 s (small-map plateau)
- threshold = 10/50/90% của steady_speed_mean
- crossing phải duy trì >= 5 sample liên tiếp
- rise_time_10_90 = t90 - t10
- encoder-equivalent accel 10-90 = 0.8*steady / rise_time
"""

import argparse
import csv
import glob
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


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
)
from action_controller_v3 import (
    ActionControllerV3,
    vehicle_planar_speed_real_mps,
)


# =============================================================================
# PROFILE - matched to REAL Test 2
# =============================================================================

CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

TARGET_SPEEDS_MPS = [0.30, 0.50, 0.70]

STOP_BEFORE_S = 3.0
ACCEL_S = 2.5
STOP_AFTER_S = 0.5

# Small-map adaptation: use the last 0.5 s as the plateau estimate.
# t10/t50/t90 remain measured from the same 0->target step.
STEADY_WINDOW_S = 0.5
CONFIRM_SAMPLES = 5

SETTLE_S = 1.0
SPAWN_Z_OFFSET_M = 0.25

EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"


# =============================================================================
# Helpers
# =============================================================================

def mean(values):
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def std_population(values):
    if not values:
        return float("nan")

    m = mean(values)

    return float(
        math.sqrt(
            sum((float(x) - m) ** 2 for x in values)
            / len(values)
        )
    )


def ensure_output_dirs():
    path = (
        PROJECT_ROOT
        / "dynamics_calibration"
        / "sim_logs"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def apply_longitudinal_tune_v2b(vehicle):
    """
    Tune V2B:
    - torque scale = 0.80
    - max brake torque = 560
    - KEEP autobox = True
    - gear_switch_time = 0.0
    - NO manual gear forcing

    Verify-physics test đã xác nhận các field này giữ nguyên sau world.tick().
    """
    physics = vehicle.get_physics_control()

    physics.torque_curve = [
        carla.Vector2D(
            float(point.x),
            float(point.y) * 0.80,
        )
        for point in list(physics.torque_curve)
    ]

    wheels = list(physics.wheels)
    for wheel in wheels:
        wheel.max_brake_torque = 560.0
    physics.wheels = wheels

    physics.use_gear_autobox = True
    physics.gear_switch_time = 0.0

    vehicle.apply_physics_control(physics)

    applied = vehicle.get_physics_control()
    return {
        "torque_curve": [
            (float(p.x), float(p.y))
            for p in list(applied.torque_curve)
        ],
        "brake": [
            float(w.max_brake_torque)
            for w in list(applied.wheels)
        ],
        "autobox": bool(applied.use_gear_autobox),
        "gear_switch_time_s": float(applied.gear_switch_time),
    }



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
            "Vehicle chỉ có {} wheels.".format(
                len(wheels)
            )
        )

    for wheel in wheels[:4]:
        wheel.radius = float(
            CARLA_WHEEL_RADIUS_CM
        )

    wheels[0].max_steer_angle = float(
        CARLA_FRONT_MAX_STEER_DEG
    )
    wheels[1].max_steer_angle = float(
        CARLA_FRONT_MAX_STEER_DEG
    )
    wheels[2].max_steer_angle = float(
        CARLA_REAR_MAX_STEER_DEG
    )
    wheels[3].max_steer_angle = float(
        CARLA_REAR_MAX_STEER_DEG
    )

    physics.wheels = wheels
    vehicle.apply_physics_control(physics)


def get_longitudinal_accel_real_mps2(vehicle):
    acc = vehicle.get_acceleration()
    forward = vehicle.get_transform().get_forward_vector()

    long_acc_carla = (
        float(acc.x) * float(forward.x)
        + float(acc.y) * float(forward.y)
        + float(acc.z) * float(forward.z)
    )

    return float(
        long_acc_carla
        / float(CARLA_GEOMETRY_SCALE)
    )


def get_yaw_rate_rad_s(vehicle):
    angular = vehicle.get_angular_velocity()
    return float(
        math.radians(float(angular.z))
    )


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

    transform = carla.Transform(
        carla.Location(
            x=float(spawn_transform.location.x),
            y=float(spawn_transform.location.y),
            z=float(spawn_transform.location.z)
            + SPAWN_Z_OFFSET_M,
        ),
        spawn_transform.rotation,
    )

    vehicle.set_transform(transform)
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

    matches = (
        world.get_blueprint_library()
        .filter(VEHICLE_BLUEPRINT_ID)
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

    apply_vehicle_geometry(vehicle)
    tune = apply_longitudinal_tune_v2b(vehicle)
    world.tick()
    verify = vehicle.get_physics_control()
    print(
        "    TUNE V2B | torque={} | brake={} | autobox={} | gear_switch={:.3f}s".format(
            tune["torque_curve"],
            tune["brake"],
            bool(verify.use_gear_autobox),
            float(verify.gear_switch_time),
        )
    )
    world.tick()

    return vehicle, original


def sample_row(
    segment,
    segment_elapsed_s,
    total_elapsed_s,
    speed_cmd_mps,
    vehicle,
    control_info,
    sim_frame,
):
    control = vehicle.get_control()

    return {
        "segment": segment,
        "jetson_elapsed_s": float(
            segment_elapsed_s
        ),
        "time_ms": int(
            round(total_elapsed_s * 1000.0)
        ),
        "speed_cmd_mps": float(speed_cmd_mps),
        "speed_mps": float(
            vehicle_planar_speed_real_mps(vehicle)
        ),

        # REAL-schema-compatible placeholders
        "motor_pwm": "",
        "encoder_count": "",
        "encoder_delta_count": "",
        "steer_cmd_deg": 0.0,
        "steer_fb_deg": "",
        "servo_target_raw": "",
        "servo_fb_raw": "",
        "gyro_z_rad_s": float(
            get_yaw_rate_rad_s(vehicle)
        ),
        "linear_accel_x_m_s2": float(
            get_longitudinal_accel_real_mps2(
                vehicle
            )
        ),
        "yaw_model_rad_s": "",
        "servo_valid": "",
        "imu_valid": "",
        "failsafe": 0,

        # CARLA diagnostics
        "sim_frame": int(sim_frame),
        "requested_speed_cmd_mps": float(
            control_info[
                "requested_speed_cmd_mps"
            ]
        ),
        "applied_speed_cmd_mps": float(
            control_info[
                "applied_speed_cmd_mps"
            ]
        ),
        "carla_speed_raw_mps": float(
            control_info[
                "current_speed_carla_mps"
            ]
        ),
        "target_speed_carla_mps": float(
            control_info[
                "target_speed_carla_mps"
            ]
        ),
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "carla_steer": float(control.steer),
        "gear": int(control.gear),
        "manual_gear_shift": int(bool(control.manual_gear_shift)),
    }


def run_segment(
    world,
    vehicle,
    controller,
    rows,
    segment_name,
    speed_cmd_mps,
    duration_s,
    total_elapsed_s,
):
    n_ticks = max(
        1,
        int(round(float(duration_s) * CONTROL_HZ)),
    )

    for i in range(n_ticks):
        info = controller.step(
            steer_cmd=0.0,
            speed_cmd_mps=float(speed_cmd_mps),
            dt=DT,
        )

        frame = int(world.tick())

        segment_elapsed_s = (i + 1) * DT
        total_elapsed_s += DT

        rows.append(
            sample_row(
                segment=segment_name,
                segment_elapsed_s=segment_elapsed_s,
                total_elapsed_s=total_elapsed_s,
                speed_cmd_mps=speed_cmd_mps,
                vehicle=vehicle,
                control_info=info,
                sim_frame=frame,
            )
        )

    return total_elapsed_s


def first_confirmed_crossing_ms(
    segment_rows,
    threshold_mps,
    confirm_samples=CONFIRM_SAMPLES,
):
    """
    Giống cách REAL summary đã được suy ra:
    phải >= threshold liên tục confirm_samples sample.
    Trả thời gian tính từ onset của command > 0.
    """
    if not segment_rows:
        return float("nan")

    onset_idx = None

    for i, row in enumerate(segment_rows):
        if float(row["speed_cmd_mps"]) > 0.0:
            onset_idx = i
            break

    if onset_idx is None:
        return float("nan")

    onset_time = float(
        segment_rows[onset_idx][
            "jetson_elapsed_s"
        ]
    )

    values = [
        float(row["speed_mps"])
        for row in segment_rows
    ]

    for i in range(
        onset_idx,
        len(segment_rows)
        - confirm_samples
        + 1,
    ):
        window = values[
            i:i + confirm_samples
        ]

        if all(
            value >= float(threshold_mps)
            for value in window
        ):
            crossing_time = float(
                segment_rows[i][
                    "jetson_elapsed_s"
                ]
            )

            return float(
                (crossing_time - onset_time)
                * 1000.0
            )

    return float("nan")


def rows_in_time_range(
    rows,
    start_s,
    end_s,
):
    return [
        row
        for row in rows
        if (
            float(row["jetson_elapsed_s"])
            >= float(start_s)
            and float(row["jetson_elapsed_s"])
            <= float(end_s)
        )
    ]


def build_summary(rows):
    summary = []

    for target in TARGET_SPEEDS_MPS:
        segment_name = "accel_0_to_{:.2f}".format(
            target
        )

        segment_rows = [
            row
            for row in rows
            if row["segment"] == segment_name
        ]

        if not segment_rows:
            continue

        steady_start = (
            ACCEL_S - STEADY_WINDOW_S
        )

        steady_rows = [
            row
            for row in segment_rows
            if float(
                row["jetson_elapsed_s"]
            ) >= steady_start
        ]

        steady_speeds = [
            float(row["speed_mps"])
            for row in steady_rows
        ]

        steady_mean = mean(
            steady_speeds
        )

        steady_std = std_population(
            steady_speeds
        )

        threshold_10 = (
            0.10 * steady_mean
        )
        threshold_50 = (
            0.50 * steady_mean
        )
        threshold_90 = (
            0.90 * steady_mean
        )

        t10_ms = first_confirmed_crossing_ms(
            segment_rows,
            threshold_10,
        )

        t50_ms = first_confirmed_crossing_ms(
            segment_rows,
            threshold_50,
        )

        t90_ms = first_confirmed_crossing_ms(
            segment_rows,
            threshold_90,
        )

        if (
            math.isfinite(t10_ms)
            and math.isfinite(t90_ms)
        ):
            rise_ms = (
                t90_ms - t10_ms
            )
        else:
            rise_ms = float("nan")

        if (
            math.isfinite(rise_ms)
            and rise_ms > 0.0
        ):
            encoder_equiv_accel = (
                0.80 * steady_mean
                / (rise_ms / 1000.0)
            )
        else:
            encoder_equiv_accel = (
                float("nan")
            )

        # CARLA physics accel, chỉ dùng phụ trợ.
        if (
            math.isfinite(t10_ms)
            and math.isfinite(t90_ms)
        ):
            onset_time = float(
                segment_rows[0][
                    "jetson_elapsed_s"
                ]
            )

            accel_window_start = (
                onset_time
                + t10_ms / 1000.0
            )

            accel_window_end = (
                onset_time
                + t90_ms / 1000.0
            )

            accel_rows = rows_in_time_range(
                segment_rows,
                accel_window_start,
                accel_window_end,
            )
        else:
            accel_rows = []

        accel_values = [
            float(
                row[
                    "linear_accel_x_m_s2"
                ]
            )
            for row in accel_rows
        ]

        throttle_values = [
            float(row["throttle"])
            for row in segment_rows
        ]

        summary.append({
            "target_speed_mps": float(
                target
            ),
            "steady_speed_mean_mps": (
                steady_mean
            ),
            "steady_speed_std_mps": (
                steady_std
            ),
            "time_to_10_ms": t10_ms,
            "time_to_50_ms": t50_ms,
            "time_to_90_ms": t90_ms,
            "rise_time_10_90_ms": rise_ms,
            "encoder_accel_10_90_m_s2": (
                encoder_equiv_accel
            ),

            # CARLA's physics-derived longitudinal accel.
            # Không coi là tương đương raw IMU REAL.
            "carla_long_accel_mean_10_90_m_s2": (
                mean(accel_values)
            ),
            "carla_long_accel_std_10_90_m_s2": (
                std_population(accel_values)
            ),
            "carla_long_accel_max_10_90_m_s2": (
                max(accel_values)
                if accel_values
                else float("nan")
            ),

            "throttle_mean_full_step": (
                mean(throttle_values)
            ),
            "confirm_samples": (
                CONFIRM_SAMPLES
            ),
            "samples": len(
                segment_rows
            ),
        })

    return summary


def find_latest_real_summary():
    pattern = str(
        PROJECT_ROOT
        / "Log"
        / "test2_acceleration_*_summary.csv"
    )

    candidates = glob.glob(pattern)

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: os.path.getmtime(p)
    )

    return Path(candidates[-1])


def read_csv_rows(path):
    with open(
        str(path),
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def build_real_sim_comparison(
    real_summary_path,
    sim_summary,
):
    if real_summary_path is None:
        return []

    real_rows = read_csv_rows(
        real_summary_path
    )

    real_by_target = {}

    for row in real_rows:
        try:
            target = round(
                float(row["target_speed_mps"]),
                3,
            )

            real_by_target[target] = row
        except Exception:
            continue

    comparison = []

    for sim in sim_summary:
        target = round(
            float(sim["target_speed_mps"]),
            3,
        )

        real = real_by_target.get(target)

        if real is None:
            continue

        real_steady = float(
            real["steady_speed_mean_mps"]
        )

        real_rise = float(
            real["rise_time_10_90_ms"]
        )

        sim_steady = float(
            sim["steady_speed_mean_mps"]
        )

        sim_rise = float(
            sim["rise_time_10_90_ms"]
        )

        comparison.append({
            "target_speed_mps": target,

            "real_steady_speed_mps": (
                real_steady
            ),
            "sim_steady_speed_mps": (
                sim_steady
            ),
            "steady_sim_minus_real_mps": (
                sim_steady - real_steady
            ),
            "steady_error_pct": (
                (sim_steady - real_steady)
                / real_steady
                * 100.0
                if abs(real_steady) > 1e-12
                else float("nan")
            ),

            "real_time_to_10_ms": float(
                real["time_to_10_ms"]
            ),
            "sim_time_to_10_ms": float(
                sim["time_to_10_ms"]
            ),

            "real_time_to_50_ms": float(
                real["time_to_50_ms"]
            ),
            "sim_time_to_50_ms": float(
                sim["time_to_50_ms"]
            ),

            "real_time_to_90_ms": float(
                real["time_to_90_ms"]
            ),
            "sim_time_to_90_ms": float(
                sim["time_to_90_ms"]
            ),

            "real_rise_time_10_90_ms": (
                real_rise
            ),
            "sim_rise_time_10_90_ms": (
                sim_rise
            ),
            "rise_time_error_ms": (
                sim_rise - real_rise
            ),
            "rise_time_error_pct": (
                (sim_rise - real_rise)
                / real_rise
                * 100.0
                if abs(real_rise) > 1e-12
                else float("nan")
            ),

            "real_encoder_accel_10_90_m_s2": float(
                real[
                    "encoder_accel_10_90_m_s2"
                ]
            ),
            "sim_encoder_equiv_accel_10_90_m_s2": float(
                sim[
                    "encoder_accel_10_90_m_s2"
                ]
            ),

            # REAL IMU shown for reference only.
            "real_imu_accel_corrected_m_s2": float(
                real[
                    "imu_accel_mean_10_90_corrected_m_s2"
                ]
            ),
            "sim_carla_long_accel_mean_10_90_m_s2": float(
                sim[
                    "carla_long_accel_mean_10_90_m_s2"
                ]
            ),
        })

    return comparison


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(
            "Không có dữ liệu để ghi: {}".format(
                path
            )
        )

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


def print_summary(
    sim_summary,
    comparison,
):
    print()
    print("=" * 112)
    print("GĐ4 TEST 2 - ACCELERATION RESULT")
    print("=" * 112)

    if comparison:
        print(
            "{:<8} {:>10} {:>10} {:>9} | {:>10} {:>10} {:>10} {:>9}".format(
                "target",
                "REAL v",
                "SIM v",
                "v err%",
                "REAL rise",
                "SIM rise",
                "Δrise",
                "rise%",
            )
        )

        for row in comparison:
            print(
                "{:<8.2f} {:>10.4f} {:>10.4f} {:>+8.2f}% | {:>8.0f}ms {:>8.0f}ms {:>+8.0f}ms {:>+8.2f}%".format(
                    float(row["target_speed_mps"]),
                    float(row["real_steady_speed_mps"]),
                    float(row["sim_steady_speed_mps"]),
                    float(row["steady_error_pct"]),
                    float(row["real_rise_time_10_90_ms"]),
                    float(row["sim_rise_time_10_90_ms"]),
                    float(row["rise_time_error_ms"]),
                    float(row["rise_time_error_pct"]),
                )
            )
    else:
        print(
            "{:<8} {:>12} {:>12} {:>12} {:>12}".format(
                "target",
                "SIM steady",
                "t10 ms",
                "t90 ms",
                "rise ms",
            )
        )

        for row in sim_summary:
            print(
                "{:<8.2f} {:>12.4f} {:>12.0f} {:>12.0f} {:>12.0f}".format(
                    float(row["target_speed_mps"]),
                    float(row["steady_speed_mean_mps"]),
                    float(row["time_to_10_ms"]),
                    float(row["time_to_90_ms"]),
                    float(row["rise_time_10_90_ms"]),
                )
            )

    print("=" * 112)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "GĐ4 CARLA Test 2 - Acceleration Step"
        )
    )

    parser.add_argument(
        "--host",
        default=HOST,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--spawn",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--allow-other-map",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = ensure_output_dirs()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_path = (
        output_dir
        / (
            "test2_acceleration_carla_"
            + stamp
            + ".csv"
        )
    )

    summary_path = (
        output_dir
        / (
            "test2_acceleration_carla_"
            + stamp
            + "_summary.csv"
        )
    )

    compare_path = (
        output_dir
        / (
            "test2_acceleration_real_vs_carla_"
            + stamp
            + ".csv"
        )
    )

    client = carla.Client(
        args.host,
        args.port,
    )
    client.set_timeout(
        float(args.timeout)
    )

    world = client.get_world()
    map_name = world.get_map().name

    print("=" * 88)
    print("GĐ4 - TEST 2 ACCELERATION STEP - SMALL MAP - TUNE V2B AUTOBOX ZERO-SHIFT")
    print("=" * 88)
    print("MAP       :", map_name)
    print("VEHICLE   :", VEHICLE_BLUEPRINT_ID)
    print("CONTROL   : {:.1f} Hz | dt={:.3f} s".format(
        CONTROL_HZ,
        DT,
    ))
    print("SCALE     : {:.1f}x".format(
        CARLA_GEOMETRY_SCALE
    ))
    print("SPAWN     :", args.spawn)
    print("TARGETS   :", TARGET_SPEEDS_MPS)
    print("=" * 88)

    if (
        EXPECTED_MAP_FRAGMENT.lower()
        not in map_name.lower()
        and not args.allow_other_map
    ):
        raise RuntimeError(
            "Sai map: '{}'. Cần map chứa '{}'.".format(
                map_name,
                EXPECTED_MAP_FRAGMENT,
            )
        )

    original_settings = world.get_settings()

    vehicle = None
    rows = []
    total_elapsed_s = 0.0

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        world.tick()

        vehicle, spawn_transform = spawn_vehicle(
            world,
            args.spawn,
        )

        controller = ActionControllerV3(
            vehicle=vehicle,
            steer_rate_limit_per_s=None,
            speed_rate_limit_mps2=None,
        )

        real_summary_path = (
            find_latest_real_summary()
        )

        if real_summary_path is not None:
            print("REAL REF   :", real_summary_path)
        else:
            print(
                "REAL REF   : chưa tìm thấy "
                "Log/test2_acceleration_*_summary.csv"
            )

        for target in TARGET_SPEEDS_MPS:
            print()
            print(
                ">>> TEST 0 -> {:.2f} m/s".format(
                    target
                )
            )

            reset_vehicle_to_spawn(
                world,
                vehicle,
                spawn_transform,
            )

            controller.reset()
            world.tick()

            total_elapsed_s = run_segment(
                world=world,
                vehicle=vehicle,
                controller=controller,
                rows=rows,
                segment_name=(
                    "stop_before_{:.2f}".format(
                        target
                    )
                ),
                speed_cmd_mps=0.0,
                duration_s=STOP_BEFORE_S,
                total_elapsed_s=total_elapsed_s,
            )

            total_elapsed_s = run_segment(
                world=world,
                vehicle=vehicle,
                controller=controller,
                rows=rows,
                segment_name=(
                    "accel_0_to_{:.2f}".format(
                        target
                    )
                ),
                speed_cmd_mps=target,
                duration_s=ACCEL_S,
                total_elapsed_s=total_elapsed_s,
            )

            total_elapsed_s = run_segment(
                world=world,
                vehicle=vehicle,
                controller=controller,
                rows=rows,
                segment_name=(
                    "stop_after_{:.2f}".format(
                        target
                    )
                ),
                speed_cmd_mps=0.0,
                duration_s=STOP_AFTER_S,
                total_elapsed_s=total_elapsed_s,
            )

            current_summary = build_summary(
                rows
            )

            current = [
                row
                for row in current_summary
                if abs(
                    float(row["target_speed_mps"])
                    - target
                ) < 1e-9
            ][0]

            print(
                "    SIM steady={:.4f} m/s | "
                "t10={:.0f} ms | t50={:.0f} ms | "
                "t90={:.0f} ms | rise10-90={:.0f} ms".format(
                    float(
                        current[
                            "steady_speed_mean_mps"
                        ]
                    ),
                    float(
                        current["time_to_10_ms"]
                    ),
                    float(
                        current["time_to_50_ms"]
                    ),
                    float(
                        current["time_to_90_ms"]
                    ),
                    float(
                        current[
                            "rise_time_10_90_ms"
                        ]
                    ),
                )
            )

        sim_summary = build_summary(rows)

        comparison = (
            build_real_sim_comparison(
                real_summary_path,
                sim_summary,
            )
        )

        write_csv(raw_path, rows)
        write_csv(summary_path, sim_summary)

        if comparison:
            write_csv(
                compare_path,
                comparison,
            )

        print_summary(
            sim_summary,
            comparison,
        )

        print()
        print("RAW CSV     :", raw_path)
        print("SUMMARY CSV :", summary_path)

        if comparison:
            print("COMPARE CSV :", compare_path)

        print()
        print(
            "HOÀN TẤT TEST 2. "
            "Chưa tune controller/physics cho tới khi "
            "thu xong baseline Test 1 + 2 + 3."
        )

    finally:
        if vehicle is not None:
            try:
                stop_vehicle(vehicle)
                world.tick()
            except Exception:
                pass

            try:
                if vehicle.is_alive:
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
