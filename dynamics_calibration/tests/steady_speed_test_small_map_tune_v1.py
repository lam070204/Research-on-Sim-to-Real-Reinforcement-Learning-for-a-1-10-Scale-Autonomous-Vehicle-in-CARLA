# -*- coding: utf-8 -*-
"""
GĐ4 - Test 1: CARLA Steady Speed

Mục tiêu:
- Chạy cùng profile steady-speed như REAL:
    stop_before: 2 s
    steady:      7 s
    stop_after:  2 s
- speed_cmd: 0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7 m/s
- steer_cmd = 0
- CARLA synchronous 50 Hz (dt=0.02 s) để khớp log REAL hiện tại.
- Dùng ActionControllerV3 hiện tại.
- Không PPO, không VAE, không camera.
- Xuất raw CSV + summary CSV + compare CSV (nếu tìm thấy REAL summary).

Chạy từ root project:
    python .\dynamics_calibration\tests\steady_speed_test.py

Yêu cầu:
- CARLA đã mở.
- Map custom đã load đúng trước khi chạy.
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# -----------------------------------------------------------------------------
# Cho phép chạy file trực tiếp từ dynamics_calibration/tests/
# -----------------------------------------------------------------------------

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

from dynamics_calibration.common.longitudinal_tune_v1 import (
    apply_longitudinal_tune_v1,
)


# =============================================================================
# GĐ4 TEST PROFILE
# =============================================================================

CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

SPEED_COMMANDS_MPS = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

STOP_BEFORE_S = 1.0
STEADY_S = 2.5
STOP_AFTER_S = 0.5

# REAL summary hiện dùng khoảng 4 s cuối của đoạn steady.
STEADY_ANALYSIS_WINDOW_S = 0.5

SPAWN_Z_OFFSET_M = 0.25
SETTLE_S = 1.0

EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"


# =============================================================================
# HELPERS
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


def percentile_abs_max(values):
    """
    Chỉ tiện debug. Không dùng để tune ở Test 1.
    """
    if not values:
        return float("nan")

    return max(abs(float(x)) for x in values)


def ensure_output_dirs():
    sim_log_dir = (
        PROJECT_ROOT
        / "dynamics_calibration"
        / "sim_logs"
    )

    sim_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sim_log_dir


def apply_vehicle_geometry(vehicle):
    """
    Áp đúng geometry đang dùng trong PPO V3.
    Chưa tune mass/drag/friction ở đây.
    """
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
            "Vehicle chỉ trả về {} wheels.".format(
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
    """
    Cùng convention với V3:
    CARLA vehicle acceleration chiếu lên forward vector rồi / scale.
    """
    acceleration = vehicle.get_acceleration()
    forward = vehicle.get_transform().get_forward_vector()

    accel_long_carla = (
        float(acceleration.x) * float(forward.x)
        + float(acceleration.y) * float(forward.y)
        + float(acceleration.z) * float(forward.z)
    )

    return float(
        accel_long_carla
        / float(CARLA_GEOMETRY_SCALE)
    )


def get_yaw_rate_rad_s(vehicle):
    """
    CARLA Actor.get_angular_velocity() trả deg/s.
    Chuyển z sang rad/s để cùng đơn vị REAL IMU.
    """
    angular_velocity = vehicle.get_angular_velocity()

    return float(
        math.radians(
            float(angular_velocity.z)
        )
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
        carla.Vector3D(
            x=0.0,
            y=0.0,
            z=0.0,
        )
    )

    vehicle.set_target_angular_velocity(
        carla.Vector3D(
            x=0.0,
            y=0.0,
            z=0.0,
        )
    )


def reset_vehicle_to_spawn(
    world,
    vehicle,
    spawn_transform,
    settle_seconds=SETTLE_S,
):
    """
    Đưa xe về cùng vị trí và đứng yên trước mỗi speed level.
    Không ghi đoạn settle này vào dữ liệu test.
    """
    stop_vehicle(vehicle)

    reset_transform = carla.Transform(
        carla.Location(
            x=float(spawn_transform.location.x),
            y=float(spawn_transform.location.y),
            z=float(spawn_transform.location.z)
            + SPAWN_Z_OFFSET_M,
        ),
        spawn_transform.rotation,
    )

    vehicle.set_transform(reset_transform)

    stop_vehicle(vehicle)

    n_ticks = max(
        1,
        int(round(
            float(settle_seconds)
            * CONTROL_HZ
        )),
    )

    for _ in range(n_ticks):
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


def spawn_vehicle(
    world,
    spawn_number,
):
    spawn_points = list(
        world.get_map().get_spawn_points()
    )

    if not spawn_points:
        raise RuntimeError(
            "Map không có spawn points."
        )

    spawn_index = int(spawn_number) - 1

    if (
        spawn_index < 0
        or spawn_index >= len(spawn_points)
    ):
        raise ValueError(
            "--spawn {} không hợp lệ. Map có {} spawn.".format(
                spawn_number,
                len(spawn_points),
            )
        )

    blueprint_library = (
        world.get_blueprint_library()
    )

    matches = blueprint_library.filter(
        VEHICLE_BLUEPRINT_ID
    )

    if not matches:
        raise RuntimeError(
            "Không tìm thấy blueprint {}".format(
                VEHICLE_BLUEPRINT_ID
            )
        )

    blueprint = matches[0]
    original_spawn = spawn_points[spawn_index]

    transform = carla.Transform(
        carla.Location(
            x=float(original_spawn.location.x),
            y=float(original_spawn.location.y),
            z=float(original_spawn.location.z)
            + SPAWN_Z_OFFSET_M,
        ),
        original_spawn.rotation,
    )

    vehicle = world.try_spawn_actor(
        blueprint,
        transform,
    )

    if vehicle is None:
        raise RuntimeError(
            "Không spawn được vehicle tại spawn {}. "
            "Hãy đóng vehicle/NPC khác hoặc chọn --spawn khác.".format(
                spawn_number
            )
        )

    apply_vehicle_geometry(vehicle)
    tune_info = apply_longitudinal_tune_v1(vehicle)
    world.tick()
    print(
        "    TUNE V1 | mass={:.1f} | torque={} | brake={}".format(
            tune_info["mass_kg"],
            tune_info["torque_curve"],
            tune_info["max_brake_torque"],
        )
    )

    # Commit physics.
    world.tick()

    return vehicle, original_spawn


def sample_row(
    segment,
    segment_elapsed_s,
    total_elapsed_s,
    speed_cmd_mps,
    steer_cmd,
    vehicle,
    control_info,
    sim_frame,
):
    speed_real = (
        vehicle_planar_speed_real_mps(
            vehicle
        )
    )

    accel_long = (
        get_longitudinal_accel_real_mps2(
            vehicle
        )
    )

    yaw_rate = get_yaw_rate_rad_s(
        vehicle
    )

    control = vehicle.get_control()

    # Schema đầu tiên bám gần REAL log.
    # Các field REAL không tồn tại trong CARLA được để rỗng.
    return {
        "segment": segment,
        "jetson_elapsed_s": segment_elapsed_s,
        "time_ms": int(
            round(total_elapsed_s * 1000.0)
        ),
        "speed_cmd_mps": float(
            speed_cmd_mps
        ),
        "speed_mps": speed_real,
        "motor_pwm": "",
        "encoder_count": "",
        "encoder_delta_count": "",
        "steer_cmd_deg": float(
            steer_cmd
            * CARLA_FRONT_MAX_STEER_DEG
        ),
        "steer_fb_deg": "",
        "servo_target_raw": "",
        "servo_fb_raw": "",
        "gyro_z_rad_s": yaw_rate,
        "linear_accel_x_m_s2": accel_long,
        "yaw_model_rad_s": "",
        "servo_valid": "",
        "imu_valid": "",
        "failsafe": 0,

        # CARLA-specific fields
        "sim_frame": int(sim_frame),
        "requested_steer_cmd": float(
            control_info[
                "requested_steer_cmd"
            ]
        ),
        "requested_speed_cmd_mps": float(
            control_info[
                "requested_speed_cmd_mps"
            ]
        ),
        "applied_steer_cmd": float(
            control_info[
                "applied_steer_cmd"
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
        "throttle": float(
            control.throttle
        ),
        "brake": float(
            control.brake
        ),
        "carla_steer": float(
            control.steer
        ),
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
    steer_cmd = 0.0

    n_ticks = max(
        1,
        int(round(
            float(duration_s)
            * CONTROL_HZ
        )),
    )

    for tick_index in range(n_ticks):
        control_info = controller.step(
            steer_cmd=steer_cmd,
            speed_cmd_mps=speed_cmd_mps,
            dt=DT,
        )

        sim_frame = int(
            world.tick()
        )

        segment_elapsed_s = (
            (tick_index + 1)
            * DT
        )

        total_elapsed_s += DT

        rows.append(
            sample_row(
                segment=segment_name,
                segment_elapsed_s=(
                    segment_elapsed_s
                ),
                total_elapsed_s=(
                    total_elapsed_s
                ),
                speed_cmd_mps=(
                    speed_cmd_mps
                ),
                steer_cmd=steer_cmd,
                vehicle=vehicle,
                control_info=control_info,
                sim_frame=sim_frame,
            )
        )

    return total_elapsed_s


def build_summary(rows):
    output = []

    for speed_cmd in SPEED_COMMANDS_MPS:
        segment_name = "steady_{:.2f}".format(
            speed_cmd
        )

        segment_rows = [
            row
            for row in rows
            if row["segment"] == segment_name
        ]

        if not segment_rows:
            continue

        window_start = max(
            0.0,
            STEADY_S
            - STEADY_ANALYSIS_WINDOW_S,
        )

        steady_rows = [
            row
            for row in segment_rows
            if float(
                row["jetson_elapsed_s"]
            ) >= window_start
        ]

        speeds = [
            float(row["speed_mps"])
            for row in steady_rows
        ]

        throttles = [
            float(row["throttle"])
            for row in steady_rows
        ]

        brakes = [
            float(row["brake"])
            for row in steady_rows
        ]

        output.append({
            "speed_cmd_mps": float(
                speed_cmd
            ),
            "steady_speed_mean_mps": (
                mean(speeds)
            ),
            "steady_speed_std_mps": (
                std_population(speeds)
            ),

            # Giữ cột REAL để schema dễ ghép.
            "motor_pwm_mean": "",

            "samples": len(speeds),

            # CARLA-specific diagnostics.
            "throttle_mean": (
                mean(throttles)
            ),
            "brake_mean": (
                mean(brakes)
            ),
            "steady_error_to_cmd_mps": (
                mean(speeds)
                - float(speed_cmd)
            ),
            "steady_error_to_cmd_pct": (
                (
                    mean(speeds)
                    - float(speed_cmd)
                )
                / float(speed_cmd)
                * 100.0
            ),
        })

    return output


def find_latest_real_summary():
    pattern = str(
        PROJECT_ROOT
        / "Log"
        / "test1_steady_speed_*_summary.csv"
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
        return list(
            csv.DictReader(f)
        )


def build_real_sim_comparison(
    real_summary_path,
    sim_summary,
):
    if real_summary_path is None:
        return []

    real_rows = read_csv_rows(
        real_summary_path
    )

    real_by_cmd = {}

    for row in real_rows:
        try:
            cmd = round(
                float(row["speed_cmd_mps"]),
                3,
            )

            real_by_cmd[cmd] = {
                "mean": float(
                    row[
                        "steady_speed_mean_mps"
                    ]
                ),
                "std": float(
                    row[
                        "steady_speed_std_mps"
                    ]
                ),
            }
        except Exception:
            continue

    comparison = []

    for sim in sim_summary:
        cmd = round(
            float(sim["speed_cmd_mps"]),
            3,
        )

        if cmd not in real_by_cmd:
            continue

        real_mean = real_by_cmd[cmd][
            "mean"
        ]

        sim_mean = float(
            sim["steady_speed_mean_mps"]
        )

        error_mps = sim_mean - real_mean

        if abs(real_mean) > 1e-9:
            error_pct = (
                error_mps
                / real_mean
                * 100.0
            )
        else:
            error_pct = float("nan")

        comparison.append({
            "speed_cmd_mps": cmd,
            "real_steady_mean_mps": (
                real_mean
            ),
            "sim_steady_mean_mps": (
                sim_mean
            ),
            "sim_minus_real_mps": (
                error_mps
            ),
            "error_pct_of_real": (
                error_pct
            ),
            "real_steady_std_mps": (
                real_by_cmd[cmd]["std"]
            ),
            "sim_steady_std_mps": float(
                sim[
                    "steady_speed_std_mps"
                ]
            ),
            "sim_throttle_mean": float(
                sim["throttle_mean"]
            ),
        })

    return comparison


def write_csv(
    path,
    rows,
):
    if not rows:
        raise RuntimeError(
            "Không có dữ liệu để ghi: {}".format(
                path
            )
        )

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        str(path),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    sim_summary,
    comparison,
):
    print()
    print("=" * 88)
    print("GĐ4 TEST 1 - STEADY SPEED RESULT")
    print("=" * 88)

    compare_by_cmd = {
        round(
            float(row["speed_cmd_mps"]),
            3,
        ): row
        for row in comparison
    }

    if comparison:
        print(
            "{:<8} {:>11} {:>11} {:>11} {:>10} {:>10}".format(
                "cmd",
                "REAL",
                "SIM",
                "SIM-REAL",
                "err%",
                "throttle",
            )
        )

        for sim in sim_summary:
            cmd = round(
                float(
                    sim["speed_cmd_mps"]
                ),
                3,
            )

            comp = compare_by_cmd.get(cmd)

            if comp is None:
                continue

            print(
                "{:<8.2f} {:>11.4f} {:>11.4f} {:>+11.4f} {:>+9.2f}% {:>10.3f}".format(
                    cmd,
                    float(
                        comp[
                            "real_steady_mean_mps"
                        ]
                    ),
                    float(
                        comp[
                            "sim_steady_mean_mps"
                        ]
                    ),
                    float(
                        comp[
                            "sim_minus_real_mps"
                        ]
                    ),
                    float(
                        comp[
                            "error_pct_of_real"
                        ]
                    ),
                    float(
                        comp[
                            "sim_throttle_mean"
                        ]
                    ),
                )
            )
    else:
        print(
            "{:<8} {:>14} {:>14} {:>10}".format(
                "cmd",
                "SIM mean",
                "SIM std",
                "throttle",
            )
        )

        for sim in sim_summary:
            print(
                "{:<8.2f} {:>14.4f} {:>14.4f} {:>10.3f}".format(
                    float(
                        sim["speed_cmd_mps"]
                    ),
                    float(
                        sim[
                            "steady_speed_mean_mps"
                        ]
                    ),
                    float(
                        sim[
                            "steady_speed_std_mps"
                        ]
                    ),
                    float(
                        sim["throttle_mean"]
                    ),
                )
            )

    print("=" * 88)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "GĐ4 CARLA Test 1 - Steady Speed"
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
        help=(
            "Spawn number, đánh số từ 1. "
            "Mặc định 1."
        ),
    )

    parser.add_argument(
        "--allow-other-map",
        action="store_true",
        help=(
            "Cho phép chạy nếu map không phải "
            "maptrangcorao/maptuong."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    sim_log_dir = ensure_output_dirs()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_path = (
        sim_log_dir
        / (
            "test1_steady_speed_carla_"
            + timestamp
            + ".csv"
        )
    )

    summary_path = (
        sim_log_dir
        / (
            "test1_steady_speed_carla_"
            + timestamp
            + "_summary.csv"
        )
    )

    compare_path = (
        sim_log_dir
        / (
            "test1_steady_speed_real_vs_carla_"
            + timestamp
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
    print("GĐ4 - TEST 1 STEADY SPEED - TUNE V1")
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
    print("COMMANDS  :", SPEED_COMMANDS_MPS)
    print("=" * 88)

    if (
        EXPECTED_MAP_FRAGMENT.lower()
        not in map_name.lower()
        and not args.allow_other_map
    ):
        raise RuntimeError(
            "Sai map. Hiện tại='{}'. Cần map chứa '{}'. "
            "Nếu cố ý test map khác, thêm --allow-other-map.".format(
                map_name,
                EXPECTED_MAP_FRAGMENT,
            )
        )

    original_settings = (
        world.get_settings()
    )

    vehicle = None
    rows = []
    total_elapsed_s = 0.0

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT

        # GĐ4 dynamics vẫn cần physics/render world bình thường.
        settings.no_rendering_mode = False

        world.apply_settings(settings)

        # Commit settings.
        world.tick()

        vehicle, spawn_transform = (
            spawn_vehicle(
                world=world,
                spawn_number=args.spawn,
            )
        )

        controller = ActionControllerV3(
            vehicle=vehicle,

            # GĐ4 baseline:
            # không rate-limit command để REAL/SIM nhận
            # cùng profile step.
            steer_rate_limit_per_s=None,
            speed_rate_limit_mps2=None,
        )

        real_summary_path = (
            find_latest_real_summary()
        )

        if real_summary_path is not None:
            print(
                "REAL REF   : {}".format(
                    real_summary_path
                )
            )
        else:
            print(
                "REAL REF   : chưa tìm thấy "
                "Log/test1_steady_speed_*_summary.csv"
            )

        for speed_cmd in SPEED_COMMANDS_MPS:
            print()
            print(
                ">>> TEST speed_cmd={:.2f} m/s".format(
                    speed_cmd
                )
            )

            reset_vehicle_to_spawn(
                world=world,
                vehicle=vehicle,
                spawn_transform=spawn_transform,
                settle_seconds=SETTLE_S,
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
                        speed_cmd
                    )
                ),
                speed_cmd_mps=0.0,
                duration_s=STOP_BEFORE_S,
                total_elapsed_s=(
                    total_elapsed_s
                ),
            )

            total_elapsed_s = run_segment(
                world=world,
                vehicle=vehicle,
                controller=controller,
                rows=rows,
                segment_name=(
                    "steady_{:.2f}".format(
                        speed_cmd
                    )
                ),
                speed_cmd_mps=speed_cmd,
                duration_s=STEADY_S,
                total_elapsed_s=(
                    total_elapsed_s
                ),
            )

            total_elapsed_s = run_segment(
                world=world,
                vehicle=vehicle,
                controller=controller,
                rows=rows,
                segment_name=(
                    "stop_after_{:.2f}".format(
                        speed_cmd
                    )
                ),
                speed_cmd_mps=0.0,
                duration_s=STOP_AFTER_S,
                total_elapsed_s=(
                    total_elapsed_s
                ),
            )

            # In kết quả sơ bộ command này.
            segment_name = (
                "steady_{:.2f}".format(
                    speed_cmd
                )
            )

            current_segment = [
                row
                for row in rows
                if (
                    row["segment"]
                    == segment_name
                    and float(
                        row[
                            "jetson_elapsed_s"
                        ]
                    )
                    >= (
                        STEADY_S
                        - STEADY_ANALYSIS_WINDOW_S
                    )
                )
            ]

            speed_values = [
                float(row["speed_mps"])
                for row in current_segment
            ]

            throttle_values = [
                float(row["throttle"])
                for row in current_segment
            ]

            print(
                "    SIM steady={:.4f} ± {:.4f} m/s | throttle_mean={:.3f}".format(
                    mean(speed_values),
                    std_population(
                        speed_values
                    ),
                    mean(throttle_values),
                )
            )

        sim_summary = build_summary(
            rows
        )

        comparison = (
            build_real_sim_comparison(
                real_summary_path=(
                    real_summary_path
                ),
                sim_summary=sim_summary,
            )
        )

        write_csv(
            raw_path,
            rows,
        )

        write_csv(
            summary_path,
            sim_summary,
        )

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
            "HOÀN TẤT TEST 1. "
            "Chưa tune action/physics cho tới khi xem bảng REAL ↔ SIM."
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
