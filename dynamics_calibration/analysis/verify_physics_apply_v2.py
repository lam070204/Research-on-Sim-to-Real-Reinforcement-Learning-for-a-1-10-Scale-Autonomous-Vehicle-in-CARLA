# -*- coding: utf-8 -*-
"""
GĐ4 - Verify CARLA PhysicsControl apply behavior.

Mục đích:
Tách riêng xem custom CARLA 0.9.13-dirty có thực sự chấp nhận:
- torque scale 0.80
- brake torque 560
- use_gear_autobox=False
- gear_switch_time=0.0

Mỗi CASE spawn một actor mới, áp đúng một tổ hợp, đọc lại:
- ngay sau apply_physics_control()
- sau 1 world.tick()

KHÔNG chạy xe, KHÔNG train PPO.
"""

from __future__ import print_function

import argparse
import sys
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


DT = 0.02
EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"
SPAWN_Z_OFFSET_M = 0.25


def apply_project_geometry(vehicle):
    p = vehicle.get_physics_control()
    wheels = list(p.wheels)

    for w in wheels[:4]:
        w.radius = float(CARLA_WHEEL_RADIUS_CM)

    if len(wheels) >= 4:
        wheels[0].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
        wheels[1].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
        wheels[2].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)
        wheels[3].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)

    p.wheels = wheels
    vehicle.apply_physics_control(p)


def snapshot(vehicle):
    p = vehicle.get_physics_control()
    return {
        "torque": [
            (round(float(x.x), 3), round(float(x.y), 3))
            for x in list(p.torque_curve)
        ],
        "brake": [
            round(float(w.max_brake_torque), 3)
            for w in list(p.wheels)
        ],
        "autobox": bool(p.use_gear_autobox),
        "gear_switch": round(float(p.gear_switch_time), 6),
    }


def print_snapshot(label, snap):
    print("  {:<16} torque={}".format(label, snap["torque"]))
    print("  {:<16} brake={}".format("", snap["brake"]))
    print(
        "  {:<16} autobox={} | gear_switch={}".format(
            "",
            snap["autobox"],
            snap["gear_switch"],
        )
    )


def spawn_fresh(world, spawn):
    bp = world.get_blueprint_library().find(VEHICLE_BLUEPRINT_ID)

    actor = world.try_spawn_actor(
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

    if actor is None:
        raise RuntimeError("Không spawn được actor.")

    world.tick()
    apply_project_geometry(actor)
    world.tick()
    return actor


def apply_case(vehicle, torque_scale=None, brake=None, autobox=None, gear_switch=None):
    p = vehicle.get_physics_control()

    if torque_scale is not None:
        p.torque_curve = [
            carla.Vector2D(
                float(pt.x),
                float(pt.y) * float(torque_scale),
            )
            for pt in list(p.torque_curve)
        ]

    if brake is not None:
        wheels = list(p.wheels)
        for w in wheels:
            w.max_brake_torque = float(brake)
        p.wheels = wheels

    if autobox is not None:
        p.use_gear_autobox = bool(autobox)

    if gear_switch is not None:
        p.gear_switch_time = float(gear_switch)

    vehicle.apply_physics_control(p)


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

    cases = [
        (
            "A_V1_ONLY",
            dict(
                torque_scale=0.80,
                brake=560.0,
                autobox=None,
                gear_switch=None,
            ),
        ),
        (
            "B_AUTOBOX_OFF_ONLY",
            dict(
                torque_scale=None,
                brake=None,
                autobox=False,
                gear_switch=None,
            ),
        ),
        (
            "C_V1_PLUS_AUTOBOX_OFF",
            dict(
                torque_scale=0.80,
                brake=560.0,
                autobox=False,
                gear_switch=None,
            ),
        ),
        (
            "D_V1_PLUS_GEAR_SWITCH_0",
            dict(
                torque_scale=0.80,
                brake=560.0,
                autobox=None,
                gear_switch=0.0,
            ),
        ),
        (
            "E_V1_AUTOBOX_OFF_SWITCH_0",
            dict(
                torque_scale=0.80,
                brake=560.0,
                autobox=False,
                gear_switch=0.0,
            ),
        ),
    ]

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        print("=" * 92)
        print("GĐ4 - VERIFY PHYSICS APPLY")
        print("=" * 92)

        for case_name, kwargs in cases:
            vehicle = None
            try:
                vehicle = spawn_fresh(world, spawn)

                print("")
                print("CASE:", case_name)
                print_snapshot("baseline", snapshot(vehicle))

                apply_case(vehicle, **kwargs)
                print_snapshot("after apply", snapshot(vehicle))

                world.tick()
                print_snapshot("after 1 tick", snapshot(vehicle))

            finally:
                if vehicle is not None:
                    try:
                        vehicle.destroy()
                    except Exception:
                        pass
                    world.tick()

        print("")
        print("=" * 92)
        print("DONE - gửi toàn bộ output.")
        print("=" * 92)

    finally:
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
