# -*- coding: utf-8 -*-
"""
GĐ4 - Test 3: CARLA Deceleration / Coast

Matched to REAL log:
    test3_deceleration_*.csv

Small-map profile per case:
    stop_between : 1.0 s
    pre-speed    : 2.5 s
    decel        : 2.0 s

The deceleration command itself and the 50 Hz timing are unchanged.
Only the long pre/tail durations are shortened so the vehicle does not leave the map.

Cases:
    0.50 -> 0.00 m/s
    0.70 -> 0.00 m/s
    0.70 -> 0.50 m/s

Timing:
    control/world = 50 Hz
    dt            = 0.020 s
    steer_cmd     = 0

REAL reach rule reverse-engineered from the uploaded REAL log:
- 5 consecutive samples
- final 0.00: speed <= 0.024 m/s
- final > 0: abs(speed - final) <= 0.024 m/s

Initial steady speed:
- last 25 samples of the 2.5 s pre-speed segment (~0.5 s).
- This is a small-map adaptation; use only if the plateau std is acceptably small.

No PPO / VAE / RGB camera.
Uses the current ActionControllerV3 and CARLA vehicle physics.
"""

import argparse
import csv
import glob
import math
import os
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
)
from action_controller_v3 import (
    ActionControllerV3,
    vehicle_planar_speed_real_mps,
)

from dynamics_calibration.common.longitudinal_tune_v1 import (
    apply_longitudinal_tune_v1,
)


CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

CASES = [
    (0.50, 0.00),
    (0.70, 0.00),
    (0.70, 0.50),
]

STOP_BETWEEN_S = 1.0
PRE_SPEED_S = 2.5
DECEL_S = 2.0

# REAL logger's speed resolution / reach tolerance.
REACH_TOL_MPS = 0.024
CONFIRM_SAMPLES = 5

# Small-map adaptation: last 0.5 s at 50 Hz.
# Enough to verify a plateau before issuing the deceleration step.
INITIAL_STEADY_SAMPLES = 25

SETTLE_S = 1.0
SPAWN_Z_OFFSET_M = 0.25
EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"


def mean(values):
    if not values:
        return float("nan")
    return float(sum(float(x) for x in values) / len(values))


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
    path = PROJECT_ROOT / "dynamics_calibration" / "sim_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def get_longitudinal_accel_real_mps2(vehicle):
    acc = vehicle.get_acceleration()
    forward = vehicle.get_transform().get_forward_vector()

    long_acc_carla = (
        float(acc.x) * float(forward.x)
        + float(acc.y) * float(forward.y)
        + float(acc.z) * float(forward.z)
    )

    return float(long_acc_carla / float(CARLA_GEOMETRY_SCALE))


def get_yaw_rate_rad_s(vehicle):
    angular = vehicle.get_angular_velocity()
    return float(math.radians(float(angular.z)))


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


def reset_vehicle_to_spawn(world, vehicle, spawn_transform):
    stop_vehicle(vehicle)

    transform = carla.Transform(
        carla.Location(
            x=float(spawn_transform.location.x),
            y=float(spawn_transform.location.y),
            z=float(spawn_transform.location.z) + SPAWN_Z_OFFSET_M,
        ),
        spawn_transform.rotation,
    )

    vehicle.set_transform(transform)
    stop_vehicle(vehicle)

    for _ in range(max(1, int(round(SETTLE_S * CONTROL_HZ)))):
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
    spawn_points = list(world.get_map().get_spawn_points())

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

    matches = world.get_blueprint_library().filter(VEHICLE_BLUEPRINT_ID)

    if not matches:
        raise RuntimeError(
            "Không tìm thấy blueprint {}".format(VEHICLE_BLUEPRINT_ID)
        )

    original = spawn_points[idx]

    transform = carla.Transform(
        carla.Location(
            x=float(original.location.x),
            y=float(original.location.y),
            z=float(original.location.z) + SPAWN_Z_OFFSET_M,
        ),
        original.rotation,
    )

    vehicle = world.try_spawn_actor(matches[0], transform)

    if vehicle is None:
        raise RuntimeError(
            "Không spawn được vehicle tại spawn {}.".format(spawn_number)
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
        "jetson_elapsed_s": float(segment_elapsed_s),
        "time_ms": int(round(total_elapsed_s * 1000.0)),
        "speed_cmd_mps": float(speed_cmd_mps),
        "speed_mps": float(vehicle_planar_speed_real_mps(vehicle)),

        # REAL-schema-compatible placeholders
        "motor_pwm": "",
        "encoder_count": "",
        "encoder_delta_count": "",
        "steer_cmd_deg": 0.0,
        "steer_fb_deg": "",
        "servo_target_raw": "",
        "servo_fb_raw": "",
        "gyro_z_rad_s": float(get_yaw_rate_rad_s(vehicle)),
        "linear_accel_x_m_s2": float(
            get_longitudinal_accel_real_mps2(vehicle)
        ),
        "yaw_model_rad_s": "",
        "servo_valid": "",
        "imu_valid": "",
        "failsafe": 0,

        # CARLA diagnostics
        "sim_frame": int(sim_frame),
        "requested_speed_cmd_mps": float(
            control_info["requested_speed_cmd_mps"]
        ),
        "applied_speed_cmd_mps": float(
            control_info["applied_speed_cmd_mps"]
        ),
        "carla_speed_raw_mps": float(
            control_info["current_speed_carla_mps"]
        ),
        "target_speed_carla_mps": float(
            control_info["target_speed_carla_mps"]
        ),
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "carla_steer": float(control.steer),
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
    n_ticks = max(1, int(round(float(duration_s) * CONTROL_HZ)))

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


def find_reach(segment_rows, final_cmd_mps):
    if not segment_rows:
        return None, [], float("nan")

    values = [float(row["speed_mps"]) for row in segment_rows]

    def qualifies(v):
        if float(final_cmd_mps) <= 1e-9:
            return float(v) <= REACH_TOL_MPS
        return abs(float(v) - float(final_cmd_mps)) <= REACH_TOL_MPS

    for i in range(
        0,
        len(values) - CONFIRM_SAMPLES + 1,
    ):
        window = values[i:i + CONFIRM_SAMPLES]

        if all(qualifies(v) for v in window):
            # Time is relative to first sample after command change.
            reach_ms = float(i * DT * 1000.0)
            return i, segment_rows[i:i + CONFIRM_SAMPLES], reach_ms

    return None, [], float("nan")


def build_summary(rows):
    summary = []

    for initial_cmd, final_cmd in CASES:
        pre_name = "pre_{:.2f}_to_{:.2f}".format(
            initial_cmd,
            final_cmd,
        )
        decel_name = "decel_{:.2f}_to_{:.2f}".format(
            initial_cmd,
            final_cmd,
        )

        pre_rows = [
            row for row in rows
            if row["segment"] == pre_name
        ]
        decel_rows = [
            row for row in rows
            if row["segment"] == decel_name
        ]

        if not pre_rows or not decel_rows:
            continue

        steady_rows = pre_rows[-INITIAL_STEADY_SAMPLES:]
        initial_steady = mean(
            [row["speed_mps"] for row in steady_rows]
        )
        initial_steady_std = std_population(
            [row["speed_mps"] for row in steady_rows]
        )

        reach_idx, reach_rows, reach_ms = find_reach(
            decel_rows,
            final_cmd,
        )

        reached_mean = mean(
            [row["speed_mps"] for row in reach_rows]
        )

        if math.isfinite(reach_ms) and reach_ms > 0.0:
            reach_s = reach_ms / 1000.0

            encoder_accel_to_target = (
                float(final_cmd) - float(initial_steady)
            ) / reach_s

            encoder_accel_to_measured = (
                float(reached_mean) - float(initial_steady)
            ) / reach_s
        else:
            encoder_accel_to_target = float("nan")
            encoder_accel_to_measured = float("nan")

        if reach_idx is not None:
            motion_rows = decel_rows[:reach_idx + 1]
        else:
            motion_rows = decel_rows

        accel_values = [
            float(row["linear_accel_x_m_s2"])
            for row in motion_rows
        ]
        brake_values = [
            float(row["brake"])
            for row in motion_rows
        ]
        throttle_values = [
            float(row["throttle"])
            for row in motion_rows
        ]

        summary.append({
            "initial_cmd_mps": float(initial_cmd),
            "final_cmd_mps": float(final_cmd),
            "initial_steady_speed_mps": float(initial_steady),
            "initial_steady_speed_std_mps": float(initial_steady_std),
            "reached_speed_mean_mps": float(reached_mean),
            "reach_or_stop_time_ms": float(reach_ms),
            "encoder_accel_to_target_m_s2": float(
                encoder_accel_to_target
            ),
            "encoder_accel_to_measured_reach_m_s2": float(
                encoder_accel_to_measured
            ),
            "carla_long_accel_mean_motion_m_s2": mean(accel_values),
            "carla_long_accel_std_motion_m_s2": std_population(accel_values),
            "carla_long_accel_min_motion_m_s2": (
                min(accel_values) if accel_values else float("nan")
            ),
            "carla_long_accel_max_motion_m_s2": (
                max(accel_values) if accel_values else float("nan")
            ),
            "brake_mean_motion": mean(brake_values),
            "brake_max_motion": (
                max(brake_values) if brake_values else float("nan")
            ),
            "throttle_mean_motion": mean(throttle_values),
            "reach_tolerance_mps": float(REACH_TOL_MPS),
            "confirm_samples": int(CONFIRM_SAMPLES),
            "reach_samples": int(len(reach_rows)),
            "motion_samples": int(len(motion_rows)),
            "samples": int(len(decel_rows)),
        })

    return summary


def latest_real_summary():
    pattern = str(
        PROJECT_ROOT
        / "Log"
        / "test3_deceleration_*_summary.csv"
    )

    candidates = sorted(glob.glob(pattern))

    if not candidates:
        return None

    return Path(candidates[-1])


def load_csv_dicts(path):
    with open(str(path), "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def safe_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def build_compare(sim_summary, real_summary_rows):
    compare = []

    for sim in sim_summary:
        initial_cmd = float(sim["initial_cmd_mps"])
        final_cmd = float(sim["final_cmd_mps"])

        real = None

        for row in real_summary_rows:
            if (
                abs(safe_float(row, "initial_cmd_mps") - initial_cmd) < 1e-9
                and
                abs(safe_float(row, "final_cmd_mps") - final_cmd) < 1e-9
            ):
                real = row
                break

        if real is None:
            continue

        real_initial = safe_float(
            real,
            "initial_steady_speed_mps",
        )
        real_reach = safe_float(
            real,
            "reach_or_stop_time_ms",
        )
        real_accel = safe_float(
            real,
            "encoder_accel_to_measured_reach_m_s2",
        )

        sim_initial = float(
            sim["initial_steady_speed_mps"]
        )
        sim_reach = float(
            sim["reach_or_stop_time_ms"]
        )
        sim_accel = float(
            sim["encoder_accel_to_measured_reach_m_s2"]
        )

        if math.isfinite(real_reach) and real_reach != 0.0:
            reach_error_pct = (
                (sim_reach - real_reach)
                / real_reach
                * 100.0
            )
        else:
            reach_error_pct = float("nan")

        compare.append({
            "initial_cmd_mps": initial_cmd,
            "final_cmd_mps": final_cmd,
            "real_initial_steady_speed_mps": real_initial,
            "sim_initial_steady_speed_mps": sim_initial,
            "initial_steady_error_mps": sim_initial - real_initial,
            "real_reached_speed_mean_mps": safe_float(
                real,
                "reached_speed_mean_mps",
            ),
            "sim_reached_speed_mean_mps": float(
                sim["reached_speed_mean_mps"]
            ),
            "real_reach_or_stop_time_ms": real_reach,
            "sim_reach_or_stop_time_ms": sim_reach,
            "reach_time_error_ms": sim_reach - real_reach,
            "reach_time_error_pct": reach_error_pct,
            "real_encoder_accel_to_measured_reach_m_s2": real_accel,
            "sim_encoder_accel_to_measured_reach_m_s2": sim_accel,
            "real_imu_accel_mean_motion_corrected_m_s2": safe_float(
                real,
                "imu_accel_mean_motion_corrected_m_s2",
            ),
            "sim_carla_long_accel_mean_motion_m_s2": float(
                sim["carla_long_accel_mean_motion_m_s2"]
            ),
        })

    return compare


def write_csv(path, rows):
    if not rows:
        return

    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def print_result(sim_summary, compare):
    print("")
    print("=" * 112)
    print("GĐ4 TEST 3 - DECELERATION / COAST RESULT")
    print("=" * 112)

    if compare:
        print(
            "{:<12} {:<12} {:>10} {:>10} | {:>11} {:>11} {:>11} {:>9}".format(
                "initial",
                "final",
                "REAL v0",
                "SIM v0",
                "REAL time",
                "SIM time",
                "Δtime",
                "time%",
            )
        )

        for row in compare:
            print(
                "{:<12.2f} {:<12.2f} {:>10.4f} {:>10.4f} | "
                "{:>9.0f}ms {:>9.0f}ms {:>+9.0f}ms {:>+8.2f}%".format(
                    row["initial_cmd_mps"],
                    row["final_cmd_mps"],
                    row["real_initial_steady_speed_mps"],
                    row["sim_initial_steady_speed_mps"],
                    row["real_reach_or_stop_time_ms"],
                    row["sim_reach_or_stop_time_ms"],
                    row["reach_time_error_ms"],
                    row["reach_time_error_pct"],
                )
            )
    else:
        print("Không có REAL summary để compare.")
        for row in sim_summary:
            print(
                "{:.2f}->{:.2f} | v0={:.4f} | reach={:.0f} ms | accel={:.4f}".format(
                    row["initial_cmd_mps"],
                    row["final_cmd_mps"],
                    row["initial_steady_speed_mps"],
                    row["reach_or_stop_time_ms"],
                    row["encoder_accel_to_measured_reach_m_s2"],
                )
            )

    print("=" * 112)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spawn",
        type=int,
        default=1,
        help="Spawn number, 1-based.",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("GĐ4 - TEST 3 DECELERATION / COAST - SMALL MAP - TUNE V1")
    print("=" * 88)
    print("MAP       :", EXPECTED_MAP_FRAGMENT)
    print("VEHICLE   :", VEHICLE_BLUEPRINT_ID)
    print(
        "CONTROL   : {:.1f} Hz | dt={:.3f} s".format(
            CONTROL_HZ,
            DT,
        )
    )
    print("SCALE     : {:.1f}x".format(CARLA_GEOMETRY_SCALE))
    print("SPAWN     :", args.spawn)
    print("CASES     :", CASES)
    print(
        "PROFILE   : stop {:.1f}s | pre {:.1f}s | decel {:.1f}s".format(
            STOP_BETWEEN_S,
            PRE_SPEED_S,
            DECEL_S,
        )
    )
    print(
        "REACH     : ±{:.3f} m/s | {} consecutive samples".format(
            REACH_TOL_MPS,
            CONFIRM_SAMPLES,
        )
    )
    print("=" * 88)

    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)

    world = client.get_world()
    map_name = str(world.get_map().name)

    if EXPECTED_MAP_FRAGMENT not in map_name:
        raise RuntimeError(
            "Sai map: {} | cần {}".format(
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

        vehicle, spawn_transform = spawn_vehicle(
            world,
            args.spawn,
        )

        real_ref = latest_real_summary()

        if real_ref is not None:
            print("REAL REF   :", real_ref)
        else:
            print("REAL REF   : không tìm thấy")

        controller = ActionControllerV3(vehicle)

        for initial_cmd, final_cmd in CASES:
            print("")
            print(
                ">>> TEST {:.2f} -> {:.2f} m/s".format(
                    initial_cmd,
                    final_cmd,
                )
            )

            reset_vehicle_to_spawn(
                world,
                vehicle,
                spawn_transform,
            )
            controller.reset()

            total_elapsed_s = run_segment(
                world,
                vehicle,
                controller,
                rows,
                "stop_between",
                0.0,
                STOP_BETWEEN_S,
                total_elapsed_s,
            )

            total_elapsed_s = run_segment(
                world,
                vehicle,
                controller,
                rows,
                "pre_{:.2f}_to_{:.2f}".format(
                    initial_cmd,
                    final_cmd,
                ),
                initial_cmd,
                PRE_SPEED_S,
                total_elapsed_s,
            )

            total_elapsed_s = run_segment(
                world,
                vehicle,
                controller,
                rows,
                "decel_{:.2f}_to_{:.2f}".format(
                    initial_cmd,
                    final_cmd,
                ),
                final_cmd,
                DECEL_S,
                total_elapsed_s,
            )

        sim_summary = build_summary(rows)

        real_rows = (
            load_csv_dicts(real_ref)
            if real_ref is not None
            else []
        )

        compare = build_compare(
            sim_summary,
            real_rows,
        )

        for row in sim_summary:
            print(
                "    {:.2f}->{:.2f} | SIM v0={:.4f} | "
                "reach={:.0f} ms | reached={:.4f} | accel={:.4f}".format(
                    row["initial_cmd_mps"],
                    row["final_cmd_mps"],
                    row["initial_steady_speed_mps"],
                    row["reach_or_stop_time_ms"],
                    row["reached_speed_mean_mps"],
                    row["encoder_accel_to_measured_reach_m_s2"],
                )
            )

        print_result(
            sim_summary,
            compare,
        )

        out_dir = ensure_output_dirs()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_path = out_dir / (
            "test3_deceleration_carla_{}.csv".format(stamp)
        )
        summary_path = out_dir / (
            "test3_deceleration_carla_{}_summary.csv".format(stamp)
        )
        compare_path = out_dir / (
            "test3_deceleration_real_vs_carla_{}.csv".format(stamp)
        )

        write_csv(raw_path, rows)
        write_csv(summary_path, sim_summary)

        if compare:
            write_csv(compare_path, compare)

        print("")
        print("RAW CSV     :", raw_path)
        print("SUMMARY CSV :", summary_path)

        if compare:
            print("COMPARE CSV :", compare_path)

        print("")
        print(
            "HOÀN TẤT TEST 3. Sau Test 1+2+3 mới bắt đầu tune longitudinal."
        )

    finally:
        if vehicle is not None:
            try:
                stop_vehicle(vehicle)
            except Exception:
                pass

            try:
                vehicle.destroy()
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
