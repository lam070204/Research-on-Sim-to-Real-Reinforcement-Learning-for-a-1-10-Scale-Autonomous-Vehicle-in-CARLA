# -*- coding: utf-8 -*-
"""
GĐ4 - Dump longitudinal baseline for vehicle.ty.automav3

Mục đích:
- Đọc physics control THỰC TẾ của CARLA vehicle.
- In mass, drag, torque curve, RPM, damping, transmission, COM.
- In wheel radius, tire friction, damping, brake torque.
- In constants hiện tại của ActionControllerV3.
- KHÔNG tune / KHÔNG ghi physics / KHÔNG train PPO.

Chạy từ root:
    python .\dynamics_calibration\analysis\dump_longitudinal_baseline.py --spawn 2
"""

from __future__ import print_function

import argparse
import inspect
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from simulation.carla_connection_v3 import carla
from simulation.simulation_settings_v3 import HOST, PORT
import action_controller_v3 as action_controller
from vehicle_specs_v3 import (
    VEHICLE_BLUEPRINT_ID,
    CARLA_GEOMETRY_SCALE,
    CARLA_WHEEL_RADIUS_CM,
    CARLA_FRONT_MAX_STEER_DEG,
    CARLA_REAR_MAX_STEER_DEG,
)


EXPECTED_MAP_FRAGMENT = "maptrangcorao/maptuong"
SPAWN_Z_OFFSET_M = 0.25


def get_attr(obj, name, default="<NA>"):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def fmt_vec3(v):
    if v is None or v == "<NA>":
        return str(v)
    return "({:.6f}, {:.6f}, {:.6f})".format(
        float(v.x),
        float(v.y),
        float(v.z),
    )


def fmt_curve(curve):
    if curve is None or curve == "<NA>":
        return str(curve)

    items = []
    try:
        for p in curve:
            items.append(
                "({:.3f}, {:.3f})".format(
                    float(p.x),
                    float(p.y),
                )
            )
    except Exception:
        return str(curve)

    return "[" + ", ".join(items) + "]"


def apply_project_wheel_geometry(vehicle):
    """
    Chỉ áp geometry giống Test 1/2/3:
    radius + steer angle.
    Không sửa mass/torque/brake/friction.
    """
    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)

    if len(wheels) >= 4:
        for wheel in wheels[:4]:
            wheel.radius = float(CARLA_WHEEL_RADIUS_CM)

        wheels[0].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
        wheels[1].max_steer_angle = float(CARLA_FRONT_MAX_STEER_DEG)
        wheels[2].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)
        wheels[3].max_steer_angle = float(CARLA_REAR_MAX_STEER_DEG)

        physics.wheels = wheels
        vehicle.apply_physics_control(physics)


def print_physics(vehicle):
    p = vehicle.get_physics_control()

    print("")
    print("=" * 88)
    print("CARLA PHYSICS CONTROL - CURRENT EFFECTIVE VALUES")
    print("=" * 88)

    names = [
        "mass",
        "drag_coefficient",
        "max_rpm",
        "moi",
        "damping_rate_full_throttle",
        "damping_rate_zero_throttle_clutch_engaged",
        "damping_rate_zero_throttle_clutch_disengaged",
        "use_gear_autobox",
        "gear_switch_time",
        "clutch_strength",
    ]

    for name in names:
        print("{:<46}: {}".format(name, get_attr(p, name)))

    print("{:<46}: {}".format(
        "center_of_mass",
        fmt_vec3(get_attr(p, "center_of_mass", None)),
    ))

    print("{:<46}: {}".format(
        "torque_curve",
        fmt_curve(get_attr(p, "torque_curve", None)),
    ))

    print("{:<46}: {}".format(
        "steering_curve",
        fmt_curve(get_attr(p, "steering_curve", None)),
    ))

    forward_gears = get_attr(p, "forward_gears", None)
    if forward_gears not in (None, "<NA>"):
        try:
            print("")
            print("FORWARD GEARS")
            for i, gear in enumerate(forward_gears):
                print(
                    "  gear {:02d}: ratio={} down_ratio={} up_ratio={}".format(
                        i + 1,
                        get_attr(gear, "ratio"),
                        get_attr(gear, "down_ratio"),
                        get_attr(gear, "up_ratio"),
                    )
                )
        except Exception:
            print("forward_gears:", forward_gears)

    print("")
    print("WHEELS")
    for i, wheel in enumerate(list(p.wheels)):
        side = ["front-left", "front-right", "rear-left", "rear-right"]
        label = side[i] if i < len(side) else "wheel-{}".format(i)

        print("")
        print("  [{}] {}".format(i, label))

        wheel_attrs = [
            "radius",
            "tire_friction",
            "damping_rate",
            "max_steer_angle",
            "max_brake_torque",
            "max_handbrake_torque",
        ]

        for name in wheel_attrs:
            print(
                "    {:<28}: {}".format(
                    name,
                    get_attr(wheel, name),
                )
            )

        pos = get_attr(wheel, "position", None)
        if pos not in (None, "<NA>"):
            print("    {:<28}: {}".format("position", fmt_vec3(pos)))


def print_action_controller():
    print("")
    print("=" * 88)
    print("ACTION CONTROLLER V3 - CURRENT CONSTANTS")
    print("=" * 88)

    wanted = [
        "DEFAULT_STEER_RATE_LIMIT_PER_S",
        "DEFAULT_SPEED_RATE_LIMIT_MPS2",
        "DEFAULT_FF_OFFSET",
        "DEFAULT_FF_SLOPE",
        "DEFAULT_KP_THROTTLE",
        "DEFAULT_KI_THROTTLE",
        "DEFAULT_KP_BRAKE",
        "DEFAULT_INTEGRAL_LIMIT",
        "DEFAULT_BRAKE_DEADBAND_REAL_MPS",
        "DEFAULT_INTEGRAL_DECAY_OVERSPEED",
        "DEFAULT_TARGET_DROP_RESET_MPS",
        "DEFAULT_STOP_BRAKE",
    ]

    for name in wanted:
        print(
            "{:<46}: {}".format(
                name,
                getattr(action_controller, name, "<NA>"),
            )
        )

    cls = getattr(action_controller, "ActionControllerV3", None)
    if cls is not None:
        try:
            print("")
            print("ActionControllerV3 signature:")
            print(" ", inspect.signature(cls))
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spawn",
        type=int,
        default=2,
        help="Spawn number 1-based; chỉ dùng để spawn xe rồi đọc physics.",
    )
    args = parser.parse_args()

    print("=" * 88)
    print("GĐ4 - LONGITUDINAL BASELINE DUMP")
    print("=" * 88)
    print("VEHICLE :", VEHICLE_BLUEPRINT_ID)
    print("SCALE   :", CARLA_GEOMETRY_SCALE)
    print("SPAWN   :", args.spawn)
    print("NOTE    : chỉ đọc/in thông số, không tune physics.")
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

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.02
        world.apply_settings(settings)

        spawn_points = list(world.get_map().get_spawn_points())
        idx = int(args.spawn) - 1

        if idx < 0 or idx >= len(spawn_points):
            raise ValueError(
                "--spawn {} không hợp lệ; map có {} spawn.".format(
                    args.spawn,
                    len(spawn_points),
                )
            )

        bp = world.get_blueprint_library().find(VEHICLE_BLUEPRINT_ID)
        sp = spawn_points[idx]

        transform = carla.Transform(
            carla.Location(
                x=float(sp.location.x),
                y=float(sp.location.y),
                z=float(sp.location.z) + SPAWN_Z_OFFSET_M,
            ),
            sp.rotation,
        )

        vehicle = world.try_spawn_actor(bp, transform)

        if vehicle is None:
            raise RuntimeError("Không spawn được vehicle.")

        world.tick()

        # Match geometry used by the calibration tests.
        apply_project_wheel_geometry(vehicle)
        world.tick()

        print_physics(vehicle)
        print_action_controller()

        print("")
        print("=" * 88)
        print("DONE - gửi toàn bộ output này để bắt đầu tune longitudinal vòng 1.")
        print("=" * 88)

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
