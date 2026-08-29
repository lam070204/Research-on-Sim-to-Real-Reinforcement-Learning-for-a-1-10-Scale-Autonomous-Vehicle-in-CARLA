# -*- coding: utf-8 -*-
"""
GĐ4 - Sweep KP_BRAKE cho partial deceleration 0.70 -> 0.50

Giữ nguyên Tune V2B physics:
- torque scale = 0.80
- max_brake_torque = 560
- autobox = True
- gear_switch_time = 0.0

Chỉ sweep kp_brake của ActionControllerV3:
    0.8, 1.0, 1.2, 1.4, 1.6

REAL target:
- 0.70 -> 0.50 reach ~= 560 ms
- reach band = 0.50 +/- 0.024 m/s
- confirm = 5 samples @ 50 Hz

Mục tiêu:
- tìm kp_brake đạt band gần 560 ms
- hạn chế overshoot xuống dưới 0.476 m/s
- không đụng full-stop brake torque nữa

Chạy:
    python .\dynamics_calibration\analysis\sweep_partial_brake_kp_v2b.py --spawn 2
"""

from __future__ import print_function

import argparse
import csv
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
)
from action_controller_v3 import (
    ActionControllerV3,
    vehicle_planar_speed_real_mps,
)


CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"
SPAWN_Z_OFFSET_M = 0.25

KP_VALUES = [0.8, 1.0, 1.2, 1.4, 1.6]

STOP_BEFORE_S = 1.0
PRE_SPEED_S = 2.5
DECEL_S = 2.0

INITIAL_CMD = 0.70
FINAL_CMD = 0.50

REAL_REACH_MS = 560.0
TOL_MPS = 0.024
CONFIRM_SAMPLES = 5
PRE_WINDOW_SAMPLES = 25  # 0.5 s @ 50 Hz


def apply_project_geometry(vehicle):
    p = vehicle.get_physics_control()
    wheels = list(p.wheels)

    if len(wheels) < 4:
        raise RuntimeError("Vehicle không đủ 4 wheels.")

    for w in wheels[:4]:
        w.radius = float(CARLA_WHEEL_RADIUS_CM)

    wheels[0].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
    wheels[1].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
    wheels[2].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)
    wheels[3].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)

    p.wheels = wheels
    vehicle.apply_physics_control(p)


def apply_v2b_physics(vehicle):
    p = vehicle.get_physics_control()

    p.torque_curve = [
        carla.Vector2D(
            float(point.x),
            float(point.y) * 0.80,
        )
        for point in list(p.torque_curve)
    ]

    wheels = list(p.wheels)
    for w in wheels:
        w.max_brake_torque = 560.0
    p.wheels = wheels

    p.use_gear_autobox = True
    p.gear_switch_time = 0.0

    vehicle.apply_physics_control(p)


def hard_stop(vehicle):
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
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))


def spawn_fresh(world, spawn):
    bp = world.get_blueprint_library().find(VEHICLE_BLUEPRINT_ID)

    vehicle = world.try_spawn_actor(
        bp,
        carla.Transform(
            carla.Location(
                x=float(spawn.location.x),
                y=float(spawn.location.y),
                z=float(spawn.location.z) + SPAWN_Z_OFFSET_M,
            ),
            spawn.rotation,
        ),
    )

    if vehicle is None:
        raise RuntimeError("Không spawn được vehicle.")

    world.tick()
    apply_project_geometry(vehicle)
    apply_v2b_physics(vehicle)
    world.tick()

    return vehicle


def run_cmd(world, vehicle, controller, cmd, duration_s, phase, rows, kp):
    n = int(round(float(duration_s) * CONTROL_HZ))

    for i in range(n):
        info = controller.step(
            steer_cmd=0.0,
            speed_cmd_mps=float(cmd),
            dt=DT,
        )
        frame = int(world.tick())
        control = vehicle.get_control()

        rows.append({
            "kp_brake": float(kp),
            "phase": phase,
            "elapsed_s": float((i + 1) * DT),
            "sim_frame": int(frame),
            "speed_cmd_mps": float(cmd),
            "speed_mps": float(vehicle_planar_speed_real_mps(vehicle)),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "gear": int(control.gear),
            "manual_gear_shift": int(bool(control.manual_gear_shift)),
            "speed_error_mps": float(info.get("speed_error_real_mps", 0.0)),
        })


def mean(values):
    return sum(values) / float(len(values)) if values else float("nan")


def std(values):
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(
        sum((x - m) ** 2 for x in values) / float(len(values))
    )


def first_confirmed_band(rows):
    lower = FINAL_CMD - TOL_MPS
    upper = FINAL_CMD + TOL_MPS

    flags = [
        lower <= float(r["speed_mps"]) <= upper
        for r in rows
    ]

    for i in range(0, len(flags) - CONFIRM_SAMPLES + 1):
        if all(flags[i:i + CONFIRM_SAMPLES]):
            return float(i * DT * 1000.0), i

    return float("nan"), None


def first_band_entry(rows):
    lower = FINAL_CMD - TOL_MPS
    upper = FINAL_CMD + TOL_MPS

    for i, row in enumerate(rows):
        v = float(row["speed_mps"])
        if lower <= v <= upper:
            return float(i * DT * 1000.0), i

    return float("nan"), None


def longest_band_run(rows):
    lower = FINAL_CMD - TOL_MPS
    upper = FINAL_CMD + TOL_MPS

    best = 0
    cur = 0

    for row in rows:
        v = float(row["speed_mps"])
        ok = lower <= v <= upper
        if ok:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


def evaluate(kp, pre_rows, decel_rows):
    pre_vals = [
        float(r["speed_mps"])
        for r in pre_rows[-PRE_WINDOW_SAMPLES:]
    ]

    reach_ms, reach_idx = first_confirmed_band(decel_rows)
    entry_ms, _ = first_band_entry(decel_rows)

    vals = [float(r["speed_mps"]) for r in decel_rows]

    pre_mean = mean(pre_vals)
    pre_std = std(pre_vals)
    min_v = min(vals)
    final_v = vals[-1]
    longest = longest_band_run(decel_rows)

    undershoot = max(
        0.0,
        (FINAL_CMD - TOL_MPS) - min_v,
    )

    if math.isfinite(reach_ms):
        reach_error_ms = reach_ms - REAL_REACH_MS
        score = abs(reach_error_ms) + undershoot * 3000.0
    else:
        reach_error_ms = float("nan")
        score = 100000.0 + undershoot * 3000.0

    return {
        "kp_brake": float(kp),
        "pre_mean_mps": float(pre_mean),
        "pre_std_mps": float(pre_std),
        "first_band_entry_ms": float(entry_ms),
        "confirmed_reach_ms": float(reach_ms),
        "real_reach_ms": float(REAL_REACH_MS),
        "reach_error_ms": float(reach_error_ms),
        "min_speed_mps": float(min_v),
        "final_speed_mps": float(final_v),
        "undershoot_below_band_mps": float(undershoot),
        "longest_band_samples": int(longest),
        "score": float(score),
    }


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


def fmt_ms(value):
    if math.isfinite(float(value)):
        return "{:.0f}".format(float(value))
    return "nan"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", type=int, default=2)
    args = parser.parse_args()

    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)

    world = client.get_world()

    if EXPECTED_MAP_FRAGMENT not in str(world.get_map().name):
        raise RuntimeError("Sai map: {}".format(world.get_map().name))

    spawns = list(world.get_map().get_spawn_points())
    idx = int(args.spawn) - 1

    if idx < 0 or idx >= len(spawns):
        raise RuntimeError("Spawn không hợp lệ.")

    spawn = spawns[idx]
    original_settings = world.get_settings()

    summary = []
    raw = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        print("=" * 106)
        print("GĐ4 - SWEEP PARTIAL BRAKE KP | TUNE V2B | 0.70 -> 0.50")
        print("=" * 106)
        print("KP VALUES :", KP_VALUES)
        print("REAL      : reach={} ms | band=[{:.3f}, {:.3f}] | confirm={} samples".format(
            int(REAL_REACH_MS),
            FINAL_CMD - TOL_MPS,
            FINAL_CMD + TOL_MPS,
            CONFIRM_SAMPLES,
        ))
        print("=" * 106)

        for kp in KP_VALUES:
            vehicle = None

            try:
                vehicle = spawn_fresh(world, spawn)
                controller = ActionControllerV3(
                    vehicle,
                    kp_brake=float(kp),
                )
                controller.reset()

                # stopped preparation
                run_cmd(
                    world,
                    vehicle,
                    controller,
                    0.0,
                    STOP_BEFORE_S,
                    "stop",
                    raw,
                    kp,
                )

                pre_start = len(raw)
                run_cmd(
                    world,
                    vehicle,
                    controller,
                    INITIAL_CMD,
                    PRE_SPEED_S,
                    "pre",
                    raw,
                    kp,
                )
                pre_rows = raw[pre_start:]

                decel_start = len(raw)
                run_cmd(
                    world,
                    vehicle,
                    controller,
                    FINAL_CMD,
                    DECEL_S,
                    "decel",
                    raw,
                    kp,
                )
                decel_rows = raw[decel_start:]

                result = evaluate(
                    kp,
                    pre_rows,
                    decel_rows,
                )
                summary.append(result)

                print(
                    "kp={:.2f} | pre={:.4f}±{:.4f} | entry={}ms | "
                    "reach={}ms | min={:.4f} | final={:.4f} | "
                    "band_run={} | undershoot={:.4f}".format(
                        kp,
                        result["pre_mean_mps"],
                        result["pre_std_mps"],
                        fmt_ms(result["first_band_entry_ms"]),
                        fmt_ms(result["confirmed_reach_ms"]),
                        result["min_speed_mps"],
                        result["final_speed_mps"],
                        result["longest_band_samples"],
                        result["undershoot_below_band_mps"],
                    )
                )

            finally:
                if vehicle is not None:
                    try:
                        vehicle.destroy()
                    except Exception:
                        pass
                    world.tick()

        valid = [
            row for row in summary
            if math.isfinite(float(row["confirmed_reach_ms"]))
        ]

        print("")
        print("=" * 106)
        print("SUMMARY")
        print("=" * 106)
        print(
            "{:<8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
                "kp",
                "pre",
                "reach",
                "Δreal",
                "min_v",
                "final_v",
                "undersht",
            )
        )

        for row in summary:
            print(
                "{:<8.2f} {:>10.4f} {:>9}ms {:>9}ms {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    row["kp_brake"],
                    row["pre_mean_mps"],
                    fmt_ms(row["confirmed_reach_ms"]),
                    (
                        "{:+.0f}".format(row["reach_error_ms"])
                        if math.isfinite(row["reach_error_ms"])
                        else "nan"
                    ),
                    row["min_speed_mps"],
                    row["final_speed_mps"],
                    row["undershoot_below_band_mps"],
                )
            )

        if valid:
            best = min(valid, key=lambda row: float(row["score"]))
            print("")
            print(
                "BEST CANDIDATE BY TEMP SCORE: kp_brake={:.2f} | "
                "reach={:.0f} ms | undershoot={:.4f} m/s".format(
                    best["kp_brake"],
                    best["confirmed_reach_ms"],
                    best["undershoot_below_band_mps"],
                )
            )

        out_dir = (
            PROJECT_ROOT
            / "dynamics_calibration"
            / "sim_logs"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = out_dir / (
            "sweep_partial_brake_kp_v2b_{}_summary.csv".format(stamp)
        )
        raw_path = out_dir / (
            "sweep_partial_brake_kp_v2b_{}_raw.csv".format(stamp)
        )

        write_csv(summary_path, summary)
        write_csv(raw_path, raw)

        print("")
        print("SUMMARY CSV :", summary_path)
        print("RAW CSV     :", raw_path)

    finally:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
