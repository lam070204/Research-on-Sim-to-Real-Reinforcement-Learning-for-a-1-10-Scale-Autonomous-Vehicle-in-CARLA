# -*- coding: utf-8 -*-

import math
import time

from simulation.connection import carla
from simulation.settings import CAR_NAME


# ================================================================
# TARGET ĐÃ CHỐT TỪ SỐ ĐO XE THẬT
# ================================================================

CARLA_SCALE = 10.0

REAL_WHEELBASE_M = 0.258
REAL_WHEEL_DIAMETER_M = 0.067

TARGET_WHEELBASE_M = REAL_WHEELBASE_M * CARLA_SCALE
TARGET_WHEEL_DIAMETER_M = REAL_WHEEL_DIAMETER_M * CARLA_SCALE
TARGET_WHEEL_RADIUS_CM = TARGET_WHEEL_DIAMETER_M * 100.0 / 2.0


def distance_3d(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    )


def midpoint(a, b):
    return carla.Vector3D(
        x=(a.x + b.x) / 2.0,
        y=(a.y + b.y) / 2.0,
        z=(a.z + b.z) / 2.0,
    )


def print_vector(name, v):
    print(
        "{} = ({:.6f}, {:.6f}, {:.6f})".format(
            name,
            float(v.x),
            float(v.y),
            float(v.z),
        )
    )


def find_existing_vehicle(world):
    matches = []

    for actor in world.get_actors().filter("vehicle.*"):
        try:
            if actor.type_id == CAR_NAME:
                matches.append(actor)
        except Exception:
            pass

    if matches:
        return matches[0]

    # CAR_NAME đôi khi là wildcard/filter chứ không phải type_id đầy đủ.
    for actor in world.get_actors().filter(CAR_NAME):
        try:
            if actor.type_id.startswith("vehicle."):
                return actor
        except Exception:
            pass

    return None


def spawn_vehicle(world):
    blueprints = world.get_blueprint_library().filter(CAR_NAME)

    if not blueprints:
        raise RuntimeError(
            "Không tìm thấy vehicle blueprint theo CAR_NAME={!r}".format(
                CAR_NAME
            )
        )

    print("\nBlueprint matches:")
    for bp in blueprints:
        print("  -", bp.id)

    vehicle_bp = blueprints[0]

    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        raise RuntimeError("Map hiện tại không có spawn point.")

    for index, original in enumerate(spawn_points):
        transform = carla.Transform(
            carla.Location(
                x=original.location.x,
                y=original.location.y,
                z=original.location.z + 0.25,
            ),
            original.rotation,
        )

        vehicle = world.try_spawn_actor(
            vehicle_bp,
            transform,
        )

        if vehicle is not None:
            print(
                "\nĐã spawn xe tạm tại spawn point {}.".format(
                    index + 1
                )
            )
            return vehicle

    raise RuntimeError(
        "Không spawn được xe. "
        "Hãy dọn các actor đang chiếm spawn point rồi chạy lại."
    )


def main():
    client = carla.Client(
        "127.0.0.1",
        2000,
    )
    client.set_timeout(30.0)

    world = client.get_world()

    print("=" * 72)
    print("CHECK VEHICLE PHYSICS V3")
    print("=" * 72)
    print("MAP      :", world.get_map().name)
    print("CAR_NAME :", CAR_NAME)

    vehicle = find_existing_vehicle(world)
    spawned_by_script = False

    if vehicle is None:
        vehicle = spawn_vehicle(world)
        spawned_by_script = True
        time.sleep(2.0)
    else:
        print(
            "\nDùng vehicle đang tồn tại trong world:",
            vehicle.type_id,
            "| actor id:",
            vehicle.id,
        )

    try:
        print("\n" + "=" * 72)
        print("VEHICLE")
        print("=" * 72)

        print("type_id :", vehicle.type_id)
        print("actor id:", vehicle.id)

        transform = vehicle.get_transform()
        print_vector("location", transform.location)
        print(
            "rotation = "
            "(pitch={:.3f}, yaw={:.3f}, roll={:.3f})".format(
                transform.rotation.pitch,
                transform.rotation.yaw,
                transform.rotation.roll,
            )
        )

        bbox = vehicle.bounding_box

        print("\n" + "=" * 72)
        print("BOUNDING BOX")
        print("=" * 72)

        print_vector("bbox.location", bbox.location)
        print_vector("bbox.extent", bbox.extent)

        print(
            "full size = "
            "{:.6f} x {:.6f} x {:.6f} m".format(
                bbox.extent.x * 2.0,
                bbox.extent.y * 2.0,
                bbox.extent.z * 2.0,
            )
        )

        physics = vehicle.get_physics_control()

        print("\n" + "=" * 72)
        print("VEHICLE PHYSICS")
        print("=" * 72)

        print("mass                         :", physics.mass, "kg")
        print("drag_coefficient             :", physics.drag_coefficient)
        print("max_rpm                      :", physics.max_rpm)
        print("moi                          :", physics.moi)
        print(
            "damping_rate_full_throttle   :",
            physics.damping_rate_full_throttle,
        )
        print(
            "damping_zero_clutch_engaged  :",
            physics.damping_rate_zero_throttle_clutch_engaged,
        )
        print(
            "damping_zero_clutch_disengaged:",
            physics.damping_rate_zero_throttle_clutch_disengaged,
        )
        print("use_gear_autobox             :", physics.use_gear_autobox)
        print("gear_switch_time             :", physics.gear_switch_time)
        print("clutch_strength              :", physics.clutch_strength)
        print("final_ratio                  :", physics.final_ratio)
        print(
            "use_sweep_wheel_collision    :",
            physics.use_sweep_wheel_collision,
        )

        print_vector(
            "center_of_mass",
            physics.center_of_mass,
        )

        print("\nSteering curve:")
        for point in physics.steering_curve:
            print(
                "  speed x={:.6f} -> steer scale y={:.6f}".format(
                    point.x,
                    point.y,
                )
            )

        wheels = list(physics.wheels)

        print("\n" + "=" * 72)
        print("WHEELS")
        print("=" * 72)
        print("CARLA order: 0=FL, 1=FR, 2=RL, 3=RR")

        for index, wheel in enumerate(wheels):
            names = ["FL", "FR", "RL", "RR"]
            name = (
                names[index]
                if index < len(names)
                else str(index)
            )

            print("\nWHEEL {} ({})".format(index, name))
            print_vector("position", wheel.position)
            print("radius            :", wheel.radius, "cm")
            print(
                "diameter          :",
                wheel.radius * 2.0 / 100.0,
                "m",
            )
            print(
                "max_steer_angle   :",
                wheel.max_steer_angle,
                "deg",
            )
            print("tire_friction     :", wheel.tire_friction)
            print("damping_rate      :", wheel.damping_rate)
            print(
                "max_brake_torque  :",
                wheel.max_brake_torque,
                "N*m",
            )
            print(
                "max_handbrake     :",
                wheel.max_handbrake_torque,
                "N*m",
            )

        print("\n" + "=" * 72)
        print("GEOMETRY DERIVED FROM WHEEL POSITIONS")
        print("=" * 72)

        if len(wheels) >= 4:
            fl = wheels[0].position
            fr = wheels[1].position
            rl = wheels[2].position
            rr = wheels[3].position

            front_center = midpoint(fl, fr)
            rear_center = midpoint(rl, rr)

            wheelbase = distance_3d(
                front_center,
                rear_center,
            )
            front_track = distance_3d(fl, fr)
            rear_track = distance_3d(rl, rr)

            print_vector(
                "front axle center",
                front_center,
            )
            print_vector(
                "rear axle center",
                rear_center,
            )

            print(
                "wheelbase derived   = {:.6f} m".format(
                    wheelbase
                )
            )
            print(
                "front track derived = {:.6f} m".format(
                    front_track
                )
            )
            print(
                "rear track derived  = {:.6f} m".format(
                    rear_track
                )
            )

            print("\n" + "=" * 72)
            print("COMPARE WITH CONFIRMED REAL-CAR TARGET")
            print("=" * 72)

            print(
                "Target wheelbase CARLA = {:.6f} m".format(
                    TARGET_WHEELBASE_M
                )
            )
            print(
                "Current wheelbase      = {:.6f} m".format(
                    wheelbase
                )
            )
            print(
                "Wheelbase error        = {:+.6f} m".format(
                    wheelbase - TARGET_WHEELBASE_M
                )
            )

            current_radius = (
                float(wheels[0].radius)
                if wheels
                else float("nan")
            )

            print(
                "\nTarget wheel diameter = {:.6f} m".format(
                    TARGET_WHEEL_DIAMETER_M
                )
            )
            print(
                "Target wheel radius   = {:.3f} cm".format(
                    TARGET_WHEEL_RADIUS_CM
                )
            )
            print(
                "Current FL radius     = {:.3f} cm".format(
                    current_radius
                )
            )
            print(
                "Radius error          = {:+.3f} cm".format(
                    current_radius - TARGET_WHEEL_RADIUS_CM
                )
            )

            print(
                "\nTrack width target chưa được ghi trong "
                "bộ thông số mới nhất, nên script chỉ đo chứ "
                "chưa tự kết luận đúng/sai."
            )
        else:
            print(
                "Vehicle không trả về đủ 4 WheelPhysicsControl."
            )

        print("\n" + "=" * 72)
        print("LƯU Ý")
        print("=" * 72)
        print(
            "Script này CHỈ ĐỌC thông số, KHÔNG apply physics mới."
        )
        print(
            "Sau khi có output, mới quyết định radius, steer, mass, "
            "torque/acceleration và các thông số cần chỉnh."
        )

    finally:
        if spawned_by_script and vehicle is not None:
            try:
                vehicle.destroy()
                print("\nĐã xóa vehicle tạm do script tạo.")
            except Exception:
                pass


if __name__ == "__main__":
    main()
