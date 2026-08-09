from __future__ import print_function
import sys
import time

project_root = sys.argv[1]
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from simulation.connection import carla
    print("[OK] Imported CARLA API from simulation.connection")
except Exception as exc:
    print("[FAIL] Cannot import CARLA API:", repr(exc))
    raise SystemExit(2)

try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    waypoints = carla_map.generate_waypoints(2.0)

    print("[OK] Connected to CARLA")
    print("Map       :", carla_map.name)
    print("Spawns    :", len(spawn_points))
    print("Waypoints :", len(waypoints))

    for index, transform in enumerate(spawn_points):
        print(
            "Spawn {:02d}: x={:.3f}, y={:.3f}, z={:.3f}, pitch={:.2f}, yaw={:.2f}, roll={:.2f}".format(
                index + 1,
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.pitch,
                transform.rotation.yaw,
                transform.rotation.roll,
            )
        )
except Exception as exc:
    print("[FAIL] CARLA connection/query:", repr(exc))
    raise SystemExit(3)
