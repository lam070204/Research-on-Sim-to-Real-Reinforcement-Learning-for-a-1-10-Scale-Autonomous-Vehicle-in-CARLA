import carla
import math

HOST = "127.0.0.1"
PORT = 2000

client = carla.Client(HOST, PORT)
client.set_timeout(30.0)

world = client.get_world()
carla_map = world.get_map()
spawn_points = carla_map.get_spawn_points()

print("MAP:", carla_map.name)
print("SPAWN POINTS:", len(spawn_points))
print("=" * 70)

for spawn_index, spawn in enumerate(spawn_points):
    start_wp = carla_map.get_waypoint(
        spawn.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    current_wp = start_wp

    distance = 0.0
    max_distance = 2000.0
    step = 1.0

    visited = 0
    completed_loop = False

    while distance < max_distance:
        next_wps = current_wp.next(step)

        if not next_wps:
            print(
                f"SPAWN {spawn_index}: đường bị đứt tại "
                f"{distance:.1f} m"
            )
            break

        # Giống logic repo hiện tại: chọn nhánh đầu tiên.
        next_wp = next_wps[0]

        distance += current_wp.transform.location.distance(
            next_wp.transform.location
        )

        current_wp = next_wp
        visited += 1

        # Chỉ kiểm tra quay về start sau khi đã đi ít nhất 50 m.
        if distance > 50:
            dist_to_start = current_wp.transform.location.distance(
                start_wp.transform.location
            )

            same_lane = (
                current_wp.road_id == start_wp.road_id
                and current_wp.lane_id == start_wp.lane_id
            )

            if dist_to_start < 2.0 and same_lane:
                completed_loop = True
                break

    print(
        f"SPAWN {spawn_index}: "
        f"road={start_wp.road_id}, lane={start_wp.lane_id}, "
        f"loop≈{distance:.1f} m, "
        f"waypoints={visited}, "
        f"closed={completed_loop}"
    )