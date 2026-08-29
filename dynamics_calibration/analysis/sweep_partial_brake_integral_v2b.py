# -*- coding: utf-8 -*-
"""
GĐ4 - Sweep integral retention for partial deceleration 0.70 -> 0.50

Giữ nguyên Tune V2B physics:
- torque scale = 0.80
- max_brake_torque = 560
- autobox = True
- gear_switch_time = 0.0

Giữ kp_brake = 0.80 (best candidate từ sweep trước).

Chỉ thử cách xử lý integral khi target giảm:
A) reset như hiện tại
B) không reset, decay 0.97
C) không reset, decay 0.99
D) không reset, không decay

REAL target:
- 0.70 -> 0.50 reach ~= 560 ms
- band = 0.50 +/- 0.024 m/s
- confirm 5 samples @ 50 Hz

Chạy:
    python .\dynamics_calibration\analysis\sweep_partial_brake_integral_v2b.py --spawn 2
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

KP_BRAKE = 0.80

CONFIGS = [
    {
        "name": "A_RESET_CURRENT",
        "target_drop_reset_mps": 0.08,
        "integral_decay_overspeed": 0.97,
    },
    {
        "name": "B_KEEP_I_DECAY_097",
        "target_drop_reset_mps": 0.25,
        "integral_decay_overspeed": 0.97,
    },
    {
        "name": "C_KEEP_I_DECAY_099",
        "target_drop_reset_mps": 0.25,
        "integral_decay_overspeed": 0.99,
    },
    {
        "name": "D_KEEP_I_DECAY_100",
        "target_drop_reset_mps": 0.25,
        "integral_decay_overspeed": 1.00,
    },
]

STOP_BEFORE_S = 1.0
PRE_SPEED_S = 2.5
DECEL_S = 2.0

INITIAL_CMD = 0.70
FINAL_CMD = 0.50

REAL_REACH_MS = 560.0
TOL_MPS = 0.024
CONFIRM_SAMPLES = 5
PRE_WINDOW_SAMPLES = 25


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


def run_cmd(
    world,
    vehicle,
    controller,
    cmd,
    duration_s,
    phase,
    raw_rows,
    config_name,
):
    rows = []
    n = int(round(float(duration_s) * CONTROL_HZ))

    for i in range(n):
        info = controller.step(
            steer_cmd=0.0,
            speed_cmd_mps=float(cmd),
            dt=DT,
        )
        frame = int(world.tick())
        control = vehicle.get_control()

        row = {
            "config": config_name,
            "phase": phase,
            "elapsed_s": float((i + 1) * DT),
            "sim_frame": int(frame),
            "speed_cmd_mps": float(cmd),
            "speed_mps": float(vehicle_planar_speed_real_mps(vehicle)),
            "throttle": float(control.throttle),
            "brake": float(control.brake),
            "gear": int(control.gear),
            "speed_error_mps": float(info.get("speed_error_real_mps", 0.0)),
            "speed_error_integral": float(
                info.get("speed_error_integral", 0.0)
            ),
            "throttle_ff": float(info.get("throttle_ff", 0.0)),
        }
        rows.append(row)
        raw_rows.append(row)

    return rows


def mean(values):
    return sum(values) / float(len(values)) if values else float("nan")


def std(values):
    if not values:
        return float("nan")
    m = mean(values)
    return math.sqrt(
        sum((v - m) ** 2 for v in values) / float(len(values))
    )


def first_band_entry(rows):
    low = FINAL_CMD - TOL_MPS
    high = FINAL_CMD + TOL_MPS

    for i, row in enumerate(rows):
        v = float(row["speed_mps"])
        if low <= v <= high:
            return float(i * DT * 1000.0), i

    return float("nan"), None


def confirmed_reach(rows):
    low = FINAL_CMD - TOL_MPS
    high = FINAL_CMD + TOL_MPS

    flags = [
        low <= float(row["speed_mps"]) <= high
        for row in rows
    ]

    for i in range(0, len(flags) - CONFIRM_SAMPLES + 1):
        if all(flags[i:i + CONFIRM_SAMPLES]):
            return float(i * DT * 1000.0), i

    return float("nan"), None


def longest_band_run(rows):
    low = FINAL_CMD - TOL_MPS
    high = FINAL_CMD + TOL_MPS

    best = 0
    cur = 0

    for row in rows:
        v = float(row["speed_mps"])
        if low <= v <= high:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


def closest_index_to_min_speed(rows):
    if not rows:
        return None
    return min(
        range(len(rows)),
        key=lambda i: float(rows[i]["speed_mps"]),
    )


def evaluate(config, pre_rows, decel_rows):
    pre_tail = pre_rows[-PRE_WINDOW_SAMPLES:]
    pre_speeds = [float(r["speed_mps"]) for r in pre_tail]

    entry_ms, entry_idx = first_band_entry(decel_rows)
    reach_ms, reach_idx = confirmed_reach(decel_rows)

    speeds = [float(r["speed_mps"]) for r in decel_rows]
    min_idx = closest_index_to_min_speed(decel_rows)

    min_v = min(speeds)
    final_v = speeds[-1]
    low = FINAL_CMD - TOL_MPS
    undershoot = max(0.0, low - min_v)

    integral_pre = float(pre_rows[-1]["speed_error_integral"])
    integral_after_drop = float(decel_rows[0]["speed_error_integral"])

    integral_entry = float("nan")
    if entry_idx is not None:
        integral_entry = float(
            decel_rows[entry_idx]["speed_error_integral"]
        )

    integral_min = float("nan")
    if min_idx is not None:
        integral_min = float(
            decel_rows[min_idx]["speed_error_integral"]
        )

    throttle_entry = float("nan")
    if entry_idx is not None:
        throttle_entry = float(decel_rows[entry_idx]["throttle"])

    if math.isfinite(reach_ms):
        reach_error = reach_ms - REAL_REACH_MS
    else:
        reach_error = float("nan")

    return {
        "config": config["name"],
        "target_drop_reset_mps": float(
            config["target_drop_reset_mps"]
        ),
        "integral_decay_overspeed": float(
            config["integral_decay_overspeed"]
        ),
        "kp_brake": float(KP_BRAKE),
        "pre_mean_mps": float(mean(pre_speeds)),
        "pre_std_mps": float(std(pre_speeds)),
        "first_band_entry_ms": float(entry_ms),
        "confirmed_reach_ms": float(reach_ms),
        "reach_error_ms": float(reach_error),
        "min_speed_mps": float(min_v),
        "final_speed_mps": float(final_v),
        "undershoot_below_band_mps": float(undershoot),
        "longest_band_samples": int(longest_band_run(decel_rows)),
        "integral_pre_drop": float(integral_pre),
        "integral_after_drop": float(integral_after_drop),
        "integral_at_entry": float(integral_entry),
        "integral_at_min": float(integral_min),
        "throttle_at_entry": float(throttle_entry),
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
    return (
        "{:.0f}".format(float(value))
        if math.isfinite(float(value))
        else "nan"
    )


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

    summary_rows = []
    raw_rows = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        print("=" * 122)
        print("GĐ4 - SWEEP PARTIAL BRAKE INTEGRAL RETENTION | TUNE V2B")
        print("=" * 122)
        print("kp_brake =", KP_BRAKE)
        print(
            "REAL      = 0.70 -> 0.50 | reach={} ms | band=[{:.3f},{:.3f}]".format(
                int(REAL_REACH_MS),
                FINAL_CMD - TOL_MPS,
                FINAL_CMD + TOL_MPS,
            )
        )
        print("=" * 122)

        for config in CONFIGS:
            vehicle = None

            try:
                vehicle = spawn_fresh(world, spawn)

                controller = ActionControllerV3(
                    vehicle,
                    kp_brake=KP_BRAKE,
                    target_drop_reset_mps=float(
                        config["target_drop_reset_mps"]
                    ),
                    integral_decay_overspeed=float(
                        config["integral_decay_overspeed"]
                    ),
                )
                controller.reset()

                run_cmd(
                    world,
                    vehicle,
                    controller,
                    0.0,
                    STOP_BEFORE_S,
                    "stop",
                    raw_rows,
                    config["name"],
                )

                pre_rows = run_cmd(
                    world,
                    vehicle,
                    controller,
                    INITIAL_CMD,
                    PRE_SPEED_S,
                    "pre",
                    raw_rows,
                    config["name"],
                )

                decel_rows = run_cmd(
                    world,
                    vehicle,
                    controller,
                    FINAL_CMD,
                    DECEL_S,
                    "decel",
                    raw_rows,
                    config["name"],
                )

                result = evaluate(
                    config,
                    pre_rows,
                    decel_rows,
                )
                summary_rows.append(result)

                print(
                    "{:<22} | pre={:.4f}±{:.4f} | entry={}ms | reach={}ms | "
                    "min={:.4f} | final={:.4f} | band_run={} | "
                    "Ipre={:.4f} -> I0={:.4f} -> Ientry={:.4f}".format(
                        config["name"],
                        result["pre_mean_mps"],
                        result["pre_std_mps"],
                        fmt_ms(result["first_band_entry_ms"]),
                        fmt_ms(result["confirmed_reach_ms"]),
                        result["min_speed_mps"],
                        result["final_speed_mps"],
                        result["longest_band_samples"],
                        result["integral_pre_drop"],
                        result["integral_after_drop"],
                        result["integral_at_entry"],
                    )
                )

            finally:
                if vehicle is not None:
                    try:
                        vehicle.destroy()
                    except Exception:
                        pass
                    world.tick()

        print("")
        print("=" * 122)
        print("SUMMARY")
        print("=" * 122)
        print(
            "{:<22} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
                "config",
                "reach",
                "Δreal",
                "min_v",
                "final_v",
                "under",
                "Ientry",
                "thrEntry",
            )
        )

        for row in summary_rows:
            delta = (
                "{:+.0f}".format(row["reach_error_ms"])
                if math.isfinite(row["reach_error_ms"])
                else "nan"
            )
            print(
                "{:<22} {:>7}ms {:>7}ms {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
                    row["config"],
                    fmt_ms(row["confirmed_reach_ms"]),
                    delta,
                    row["min_speed_mps"],
                    row["final_speed_mps"],
                    row["undershoot_below_band_mps"],
                    row["integral_at_entry"],
                    row["throttle_at_entry"],
                )
            )

        out_dir = PROJECT_ROOT / "dynamics_calibration" / "sim_logs"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_path = out_dir / (
            "sweep_partial_brake_integral_v2b_{}_summary.csv".format(stamp)
        )
        raw_path = out_dir / (
            "sweep_partial_brake_integral_v2b_{}_raw.csv".format(stamp)
        )

        write_csv(summary_path, summary_rows)
        write_csv(raw_path, raw_rows)

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
