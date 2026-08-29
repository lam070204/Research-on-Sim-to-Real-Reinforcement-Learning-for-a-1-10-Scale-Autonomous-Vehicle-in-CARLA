# -*- coding: utf-8 -*-
"""
GĐ4 - Diagnose Longitudinal Tune V1

Mục tiêu:
1) Kiểm tra vì sao 0 -> 0.5 có độ trễ đầu lớn.
2) Kiểm tra vì sao 0.7 -> 0.5 ra reach=nan trong Test 3 V1.

Không thay đổi Tune V1.
Không sửa controller.
Không sửa PPO environment.

Chạy:
    python .\dynamics_calibration\analysis\diagnose_tune_v1_response.py --spawn 2
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
from dynamics_calibration.common.longitudinal_tune_v1 import (
    apply_longitudinal_tune_v1,
)


CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ

SPAWN_Z_OFFSET_M = 0.25
SETTLE_S = 1.0

ACCEL_TARGET = 0.50
ACCEL_DURATION_S = 2.5

PRE_SPEED = 0.70
PRE_DURATION_S = 2.5
DECEL_TARGET = 0.50
DECEL_DURATION_S = 2.0

TARGET_BAND_TOL = 0.024
CONFIRM_SAMPLES = 5

EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"


def apply_vehicle_geometry(vehicle):
    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)

    if len(wheels) < 4:
        raise RuntimeError("Vehicle không đủ 4 wheels.")

    for wheel in wheels[:4]:
        wheel.radius = float(CARLA_WHEEL_RADIUS_CM)

    wheels[0].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
    wheels[1].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
    wheels[2].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)
    wheels[3].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)

    physics.wheels = wheels
    vehicle.apply_physics_control(physics)


def hard_reset(world, vehicle, spawn):
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
        )
    )
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))

    vehicle.set_transform(
        carla.Transform(
            carla.Location(
                x=float(spawn.location.x),
                y=float(spawn.location.y),
                z=float(spawn.location.z) + SPAWN_Z_OFFSET_M,
            ),
            spawn.rotation,
        )
    )

    for _ in range(int(round(SETTLE_S * CONTROL_HZ))):
        world.tick()

    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=False,
        )
    )
    world.tick()


def row_from_vehicle(case, phase, elapsed_s, target, vehicle, info, frame):
    control = vehicle.get_control()

    return {
        "case": case,
        "phase": phase,
        "elapsed_s": float(elapsed_s),
        "sim_frame": int(frame),
        "target_mps": float(target),
        "speed_mps": float(vehicle_planar_speed_real_mps(vehicle)),
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "gear": int(control.gear),
        "reverse": int(bool(control.reverse)),
        "manual_gear_shift": int(bool(control.manual_gear_shift)),
        "speed_error_mps": float(info.get("speed_error_real_mps", 0.0)),
        "throttle_ff": float(info.get("throttle_ff", 0.0)),
    }


def run_target(world, vehicle, controller, case, phase, target, duration_s):
    rows = []
    n = int(round(duration_s * CONTROL_HZ))

    for i in range(n):
        info = controller.step(
            steer_cmd=0.0,
            speed_cmd_mps=float(target),
            dt=DT,
        )
        frame = int(world.tick())

        rows.append(
            row_from_vehicle(
                case=case,
                phase=phase,
                elapsed_s=(i + 1) * DT,
                target=target,
                vehicle=vehicle,
                info=info,
                frame=frame,
            )
        )

    return rows


def first_time_at_or_above(rows, threshold):
    for row in rows:
        if float(row["speed_mps"]) >= float(threshold):
            return float(row["elapsed_s"])
    return float("nan")


def longest_true_run(flags):
    best = 0
    current = 0

    for value in flags:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def first_confirmed_band(rows, target, tol, confirm_samples):
    flags = [
        abs(float(r["speed_mps"]) - float(target)) <= float(tol)
        for r in rows
    ]

    for i in range(0, len(flags) - confirm_samples + 1):
        if all(flags[i:i + confirm_samples]):
            return float(rows[i]["elapsed_s"]), i

    return float("nan"), None


def gear_transitions(rows):
    result = []
    previous = None

    for row in rows:
        gear = int(row["gear"])
        if previous is None or gear != previous:
            result.append(
                (
                    float(row["elapsed_s"]),
                    gear,
                    float(row["speed_mps"]),
                    float(row["throttle"]),
                    float(row["brake"]),
                )
            )
            previous = gear

    return result


def print_accel_diagnostic(rows):
    print("")
    print("=" * 88)
    print("DIAGNOSTIC A - 0.00 -> 0.50 m/s")
    print("=" * 88)

    thresholds = [
        ("move > 0.01", 0.01),
        ("0.05 m/s", 0.05),
        ("10% target", 0.05),
        ("50% target", 0.25),
        ("90% target", 0.45),
    ]

    for label, threshold in thresholds:
        t = first_time_at_or_above(rows, threshold)
        print("{:<28}: {:>7.0f} ms".format(label, t * 1000.0))

    print("")
    print("GEAR TRANSITIONS:")
    for t, gear, speed, throttle, brake in gear_transitions(rows):
        print(
            "  t={:>6.0f} ms | gear={:>2d} | v={:.4f} | throttle={:.3f} | brake={:.3f}".format(
                t * 1000.0,
                gear,
                speed,
                throttle,
                brake,
            )
        )

    print("")
    print("FIRST 1.20 s (every 100 ms):")
    for i, row in enumerate(rows):
        if i % 5 == 4 and float(row["elapsed_s"]) <= 1.20 + 1e-9:
            print(
                "  t={:>6.0f} | v={:.4f} | thr={:.3f} | brk={:.3f} | gear={}".format(
                    float(row["elapsed_s"]) * 1000.0,
                    float(row["speed_mps"]),
                    float(row["throttle"]),
                    float(row["brake"]),
                    int(row["gear"]),
                )
            )


def print_decel_diagnostic(rows):
    lower = DECEL_TARGET - TARGET_BAND_TOL
    upper = DECEL_TARGET + TARGET_BAND_TOL

    speeds = [float(r["speed_mps"]) for r in rows]
    flags = [
        lower <= float(r["speed_mps"]) <= upper
        for r in rows
    ]

    confirmed_t, confirmed_idx = first_confirmed_band(
        rows,
        DECEL_TARGET,
        TARGET_BAND_TOL,
        CONFIRM_SAMPLES,
    )

    first_entry = float("nan")
    for row in rows:
        if lower <= float(row["speed_mps"]) <= upper:
            first_entry = float(row["elapsed_s"])
            break

    print("")
    print("=" * 88)
    print("DIAGNOSTIC B - 0.70 -> 0.50 m/s")
    print("=" * 88)
    print(
        "target band                 : [{:.3f}, {:.3f}] m/s".format(
            lower,
            upper,
        )
    )
    print(
        "first band entry            : {:.0f} ms".format(
            first_entry * 1000.0
        )
        if math.isfinite(first_entry)
        else "first band entry            : NONE"
    )
    print(
        "5-sample confirmed reach    : {:.0f} ms".format(
            confirmed_t * 1000.0
        )
        if math.isfinite(confirmed_t)
        else "5-sample confirmed reach    : NONE"
    )
    print(
        "longest consecutive in band : {} samples = {:.0f} ms".format(
            longest_true_run(flags),
            longest_true_run(flags) * DT * 1000.0,
        )
    )
    print("min speed                    : {:.4f} m/s".format(min(speeds)))
    print("max speed                    : {:.4f} m/s".format(max(speeds)))
    print("final speed                  : {:.4f} m/s".format(speeds[-1]))

    below = [v for v in speeds if v < lower]
    print(
        "overshoot below lower band  : {}".format(
            "YES, min={:.4f}".format(min(below))
            if below
            else "NO"
        )
    )

    print("")
    print("DECEL TRACE (every 100 ms):")
    for i, row in enumerate(rows):
        if i % 5 == 4:
            print(
                "  t={:>6.0f} | v={:.4f} | thr={:.3f} | brk={:.3f} | gear={}".format(
                    float(row["elapsed_s"]) * 1000.0,
                    float(row["speed_mps"]),
                    float(row["throttle"]),
                    float(row["brake"]),
                    int(row["gear"]),
                )
            )


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", type=int, default=2)
    args = parser.parse_args()

    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)

    world = client.get_world()

    if EXPECTED_MAP_FRAGMENT not in str(world.get_map().name):
        raise RuntimeError(
            "Sai map: {}".format(world.get_map().name)
        )

    original_settings = world.get_settings()
    vehicle = None

    all_rows = []

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        spawns = list(world.get_map().get_spawn_points())
        idx = int(args.spawn) - 1

        if idx < 0 or idx >= len(spawns):
            raise RuntimeError("Spawn không hợp lệ.")

        spawn = spawns[idx]
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
        apply_vehicle_geometry(vehicle)
        tune_info = apply_longitudinal_tune_v1(vehicle)
        world.tick()

        print("=" * 88)
        print("GĐ4 - DIAGNOSE TUNE V1 RESPONSE")
        print("=" * 88)
        print("SPAWN :", args.spawn)
        print("Hz    :", CONTROL_HZ)
        print("Tune  :", tune_info)
        print("=" * 88)

        # ----------------------------------------------------------
        # A) 0 -> 0.5
        # ----------------------------------------------------------
        hard_reset(world, vehicle, spawn)
        controller = ActionControllerV3(vehicle)
        controller.reset()
        world.tick()

        accel_rows = run_target(
            world,
            vehicle,
            controller,
            case="accel_0_to_0.5",
            phase="accel",
            target=ACCEL_TARGET,
            duration_s=ACCEL_DURATION_S,
        )
        all_rows.extend(accel_rows)

        print_accel_diagnostic(accel_rows)

        # ----------------------------------------------------------
        # B) 0.7 -> 0.5
        # ----------------------------------------------------------
        hard_reset(world, vehicle, spawn)
        controller.reset()
        world.tick()

        pre_rows = run_target(
            world,
            vehicle,
            controller,
            case="decel_0.7_to_0.5",
            phase="pre",
            target=PRE_SPEED,
            duration_s=PRE_DURATION_S,
        )
        all_rows.extend(pre_rows)

        print("")
        print(
            "Pre 0.70 final/mean(last 0.5s): {:.4f} / {:.4f} m/s".format(
                float(pre_rows[-1]["speed_mps"]),
                sum(
                    float(r["speed_mps"])
                    for r in pre_rows[-25:]
                ) / 25.0,
            )
        )

        decel_rows = run_target(
            world,
            vehicle,
            controller,
            case="decel_0.7_to_0.5",
            phase="decel",
            target=DECEL_TARGET,
            duration_s=DECEL_DURATION_S,
        )
        all_rows.extend(decel_rows)

        print_decel_diagnostic(decel_rows)

        out_dir = (
            PROJECT_ROOT
            / "dynamics_calibration"
            / "sim_logs"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / (
            "diagnose_tune_v1_response_{}.csv".format(stamp)
        )
        write_csv(path, all_rows)

        print("")
        print("CSV:", path)
        print("")
        print(
            "DONE. Gửi toàn bộ terminal output để quyết định Tune V2."
        )

    finally:
        if vehicle is not None:
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
