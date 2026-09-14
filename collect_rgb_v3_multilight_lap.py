# -*- coding: utf-8 -*-
"""
GĐ8 - Thu dataset RGB V3 đa ánh sáng theo từng vòng (lap) bằng CARLA Traffic Manager Autopilot
với custom vehicle: vehicle.ty.automav3

Mục tiêu:
- Không dùng PPO/VAE khi thu dataset.
- Spawn đúng custom vehicle của bạn.
- Apply wheel physics V3:
    radius = 33.5 cm
    FL/FR max steer = 35 deg
    RL/RR max steer = 0 deg
- Bật CARLA Traffic Manager autopilot.
- Thêm steering noise nhẹ tương thích CARLA 0.9.13.
- Dùng RGBDatasetCamera trong simulation/sensors_v3.py.
- Tự tiếp tục từ số ảnh đã có trong autoencoder_rgb/dataset_v3.
- Tự respawn nếu xe đứng yên quá lâu.
- Dừng khi đủ số ảnh yêu cầu.

Ví dụ smoke-test 100 ảnh:
    python collect_rgb_v3_autopilot.py --max-images 100

Thu chính thức 16000 ảnh:
    python collect_rgb_v3_autopilot.py --max-images 16000 --save-every 3

Thu 30000 ảnh:
    python collect_rgb_v3_autopilot.py --max-images 30000 --save-every 3

Lưu ý:
- Script KHÔNG xóa dataset cũ.
- Nếu thư mục đã có ảnh, collector sẽ tiếp tục đếm từ số ảnh hiện có.
- Muốn thu lại từ đầu, hãy tự xóa autoencoder_rgb/dataset_v3 trước khi chạy.
"""

import argparse
import os
import random
import time
import traceback

from simulation.carla_connection_v3 import carla
import simulation.carla_sensors_v3 as sensors_v3

RGBDatasetCamera = sensors_v3.RGBDatasetCamera

FRONT_CAMERA_WIDTH = sensors_v3.FRONT_CAMERA_WIDTH
FRONT_CAMERA_HEIGHT = sensors_v3.FRONT_CAMERA_HEIGHT
FRONT_CAMERA_FPS = sensors_v3.FRONT_CAMERA_FPS
FRONT_CAMERA_FOV = sensors_v3.FRONT_CAMERA_FOV
FRONT_CAMERA_X = sensors_v3.FRONT_CAMERA_X
FRONT_CAMERA_Y = sensors_v3.FRONT_CAMERA_Y
FRONT_CAMERA_Z = sensors_v3.FRONT_CAMERA_Z
FRONT_CAMERA_PITCH = sensors_v3.FRONT_CAMERA_PITCH
FRONT_CAMERA_YAW = sensors_v3.FRONT_CAMERA_YAW
FRONT_CAMERA_ROLL = sensors_v3.FRONT_CAMERA_ROLL


# ================================================================
# DATASET RGB V3
# Lưu trực tiếp trong autoencoder_rgb/dataset_v3
# ================================================================

RGB_DATASET_ROOT = os.path.join(
    "autoencoder_rgb",
    "dataset_v4",
)

RGB_DATASET_TRAIN_DIR = os.path.join(
    RGB_DATASET_ROOT,
    "train",
    "rgb",
)

RGB_DATASET_TEST_DIR = os.path.join(
    RGB_DATASET_ROOT,
    "test",
    "rgb",
)

# QUAN TRỌNG:
# RGBDatasetCamera dùng biến global bên trong module sensors_v3,
# nên phải đổi path trực tiếp trong module đó.
sensors_v3.RGB_DATASET_ROOT = RGB_DATASET_ROOT
sensors_v3.RGB_DATASET_TRAIN_DIR = RGB_DATASET_TRAIN_DIR
sensors_v3.RGB_DATASET_TEST_DIR = RGB_DATASET_TEST_DIR


# ================================================================
# CUSTOM VEHICLE V3
# ================================================================

VEHICLE_BLUEPRINT_ID = "vehicle.ty.automav3"

WHEEL_RADIUS_CM = 33.5
FRONT_MAX_STEER_DEG = 35.0
REAR_MAX_STEER_DEG = 0.0

# ================================================================
# REAL <-> CARLA SPEED SCALE
#
# Xe custom trong CARLA đang dùng geometry scale = 10x:
#   wheelbase real  = 0.258 m
#   wheelbase CARLA ≈ 2.58 m
#
# Quy ước kinematic cho collector/PPO V3:
#   speed_carla = speed_real * CARLA_GEOMETRY_SCALE
#
# Vì xe thật giới hạn 1.0 m/s:
#   CARLA target = 10.0 m/s = 36.0 km/h
# ================================================================
CARLA_GEOMETRY_SCALE = 10.0
DEFAULT_REAL_TARGET_SPEED_MPS = 1.0

# ================================================================
# LIGHT STEERING NOISE FOR DATASET DIVERSITY - CARLA 0.9.13
#
# CARLA 0.9.13 của bạn không có:
#   TrafficManager.vehicle_lane_offset()
#
# Vì vậy giữ Traffic Manager làm autopilot/route follower, sau đó
# cộng một steering bias rất nhỏ vào VehicleControl.steer.
#
# FRONT_MAX_STEER_DEG = 35 deg.
# Mặc định +/-0.5 deg = khoảng +/-1.43% full steer.
#
# Có thể tự chỉnh:
#   --steer-noise-deg 0.0   : tắt
#   --steer-noise-deg 0.3   : rất nhẹ
#   --steer-noise-deg 0.5   : mặc định
#   --steer-noise-deg 1.0   : mạnh hơn
#
# Target noise đổi mỗi N giây và được low-pass để tránh giật.
# ================================================================
DEFAULT_STEER_NOISE_DEG = 0.5
DEFAULT_STEER_NOISE_UPDATE_S = 2.0
DEFAULT_STEER_NOISE_SMOOTHING = 0.20


DEFAULT_SAFE_SPAWNS = [1, 2, 3, 4]

# ================================================================
# MULTI-LIGHTING PRESETS FOR VAE ROBUSTNESS
# ================================================================

LIGHTING_PRESETS = [
    ("low_045_clear",   15.0,  45.0,  5.0),
    ("low_135_clear",   15.0, 135.0,  5.0),
    ("low_225_clear",   15.0, 225.0,  5.0),
    ("low_315_clear",   15.0, 315.0,  5.0),

    ("mid_045_clear",   30.0,  45.0, 10.0),
    ("mid_135_clear",   30.0, 135.0, 10.0),
    ("mid_225_clear",   30.0, 225.0, 10.0),
    ("mid_315_clear",   30.0, 315.0, 10.0),

    ("high_045_clear",  50.0,  45.0, 10.0),
    ("high_135_clear",  50.0, 135.0, 10.0),
    ("high_225_clear",  50.0, 225.0, 10.0),
    ("high_315_clear",  50.0, 315.0, 10.0),

    ("noon_soft",       70.0,  90.0, 35.0),
    ("noon_cloudy",     70.0, 270.0, 65.0),
    ("mid_cloudy_a",    35.0,  90.0, 55.0),
    ("mid_cloudy_b",    35.0, 270.0, 55.0),
]


def apply_lighting_preset(world, preset_index):
    """
    Đổi ánh sáng vật lý trong CARLA theo từng session.
    Chỉ thay mặt trời + mây, không thêm mưa/fog/wet road ở giai đoạn này.
    """
    preset_index = int(preset_index) % len(LIGHTING_PRESETS)
    name, sun_altitude, sun_azimuth, cloudiness = LIGHTING_PRESETS[preset_index]

    weather = world.get_weather()

    weather.sun_altitude_angle = float(sun_altitude)
    weather.sun_azimuth_angle = float(sun_azimuth)
    weather.cloudiness = float(cloudiness)

    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wetness = 0.0
    weather.fog_density = 0.0
    weather.fog_distance = 1000.0
    weather.wind_intensity = 0.0

    world.set_weather(weather)

    print(
        "LIGHTING | preset={}/{} | name={} | sun_alt={:.1f} | "
        "sun_az={:.1f} | cloud={:.1f}".format(
            preset_index,
            len(LIGHTING_PRESETS) - 1,
            name,
            sun_altitude,
            sun_azimuth,
            cloudiness,
        )
    )

    return preset_index, name


# ================================================================
# HELPERS
# ================================================================

def count_pngs(folder):
    if not os.path.isdir(folder):
        return 0

    return len([
        name
        for name in os.listdir(folder)
        if name.lower().endswith(".png")
    ])


def count_dataset_images():
    return (
        count_pngs(RGB_DATASET_TRAIN_DIR)
        + count_pngs(RGB_DATASET_TEST_DIR)
    )


def speed_mps(vehicle):
    velocity = vehicle.get_velocity()

    return (
        velocity.x ** 2
        + velocity.y ** 2
        + velocity.z ** 2
    ) ** 0.5


def location_distance(a, b):
    """
    Khoảng cách Euclidean giữa hai carla.Location.
    Đơn vị là mét trong CARLA.
    """
    dx = float(a.x) - float(b.x)
    dy = float(a.y) - float(b.y)
    dz = float(a.z) - float(b.z)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def set_autopilot_target_speed(
    traffic_manager,
    vehicle,
    carla_target_speed_mps,
    fallback_percent=90.0,
):
    """
    Đặt tốc độ CARLA mục tiêu thông qua Traffic Manager 0.9.13.

    Traffic Manager dùng:
        percentage = (1 - target_kmh / speed_limit_kmh) * 100

    percentage > 0:
        chạy chậm hơn speed limit.

    percentage < 0:
        cho phép chạy nhanh hơn speed limit.

    Ví dụ:
        target CARLA = 10 m/s = 36 km/h
        speed limit  = 30 km/h
        percentage   = -20%
    """
    target_speed_kmh = float(carla_target_speed_mps) * 3.6

    try:
        speed_limit_kmh = float(vehicle.get_speed_limit())
    except Exception:
        speed_limit_kmh = 0.0

    if speed_limit_kmh > 0.1:
        difference_percent = (
            1.0 - target_speed_kmh / speed_limit_kmh
        ) * 100.0

        # Cho phép giá trị âm để target có thể lớn hơn speed limit.
        # Giữ biên an toàn để tránh giá trị cực đoan do map lỗi.
        difference_percent = max(
            -100.0,
            min(difference_percent, 99.0),
        )
    else:
        difference_percent = max(
            -100.0,
            min(float(fallback_percent), 99.0),
        )

    traffic_manager.vehicle_percentage_speed_difference(
        vehicle,
        difference_percent,
    )

    return speed_limit_kmh, difference_percent


def clamp(value, low, high):
    return max(low, min(value, high))


def steer_deg_to_normalized(steer_deg):
    """
    CARLA VehicleControl.steer thuộc [-1, 1].

    Với vehicle.ty.automav3 hiện tại:
        |steer| = 1 ~ front max steer 35 deg.
    """
    if FRONT_MAX_STEER_DEG <= 0.0:
        return 0.0

    return clamp(
        float(steer_deg) / float(FRONT_MAX_STEER_DEG),
        -1.0,
        1.0,
    )


def apply_light_steer_noise(vehicle, noise_deg):
    """
    Giữ nguyên throttle/brake/gear của Traffic Manager,
    chỉ cộng một bias nhỏ vào steer.

    Nên gọi sau world.wait_for_tick() để lấy control mới nhất
    mà Traffic Manager vừa tạo ở tick đó.
    """
    control = vehicle.get_control()

    noise_norm = steer_deg_to_normalized(noise_deg)

    noisy_control = carla.VehicleControl(
        throttle=float(control.throttle),
        steer=clamp(
            float(control.steer) + noise_norm,
            -1.0,
            1.0,
        ),
        brake=float(control.brake),
        hand_brake=bool(control.hand_brake),
        reverse=bool(control.reverse),
        manual_gear_shift=bool(control.manual_gear_shift),
        gear=int(control.gear),
    )

    vehicle.apply_control(noisy_control)

    return noise_norm, noisy_control.steer


def apply_automav3_wheel_physics(vehicle):
    """
    Apply đúng wheel physics đã chốt từ xe thật.
    """
    if vehicle.type_id != VEHICLE_BLUEPRINT_ID:
        raise RuntimeError(
            "Sai vehicle. Expected {}, got {}".format(
                VEHICLE_BLUEPRINT_ID,
                vehicle.type_id,
            )
        )

    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)

    if len(wheels) < 4:
        raise RuntimeError(
            "{} không trả về đủ 4 wheels.".format(
                VEHICLE_BLUEPRINT_ID
            )
        )

    for wheel in wheels:
        wheel.radius = float(WHEEL_RADIUS_CM)

    wheels[0].max_steer_angle = float(FRONT_MAX_STEER_DEG)  # FL
    wheels[1].max_steer_angle = float(FRONT_MAX_STEER_DEG)  # FR
    wheels[2].max_steer_angle = float(REAR_MAX_STEER_DEG)   # RL
    wheels[3].max_steer_angle = float(REAR_MAX_STEER_DEG)   # RR

    physics.wheels = wheels
    vehicle.apply_physics_control(physics)

    applied = list(vehicle.get_physics_control().wheels)

    print("\nAUTOMAV3 WHEEL PHYSICS")
    print("-" * 70)

    names = ["FL", "FR", "RL", "RR"]

    for index in range(4):
        print(
            "{}: radius={:.3f} cm | max_steer={:.3f} deg".format(
                names[index],
                float(applied[index].radius),
                float(applied[index].max_steer_angle),
            )
        )

    print("-" * 70)


def settle_vehicle(world, vehicle, seconds=1.0):
    """
    Cho xe rơi xuống mặt đường ổn định trước khi bật autopilot.
    """
    try:
        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )

        vehicle.set_target_velocity(
            carla.Vector3D(0.0, 0.0, 0.0)
        )
        vehicle.set_target_angular_velocity(
            carla.Vector3D(0.0, 0.0, 0.0)
        )

        start = time.time()

        while time.time() - start < seconds:
            try:
                world.wait_for_tick(1.0)
            except Exception:
                time.sleep(0.05)

        vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                brake=1.0,
                hand_brake=False,
            )
        )

    except Exception as error:
        print("WARNING settle_vehicle():", error)


def build_spawn_indices(world, requested_spawn_numbers):
    spawn_points = list(world.get_map().get_spawn_points())

    if not spawn_points:
        raise RuntimeError(
            "Map hiện tại không có spawn point."
        )

    valid_indices = []

    for spawn_number in requested_spawn_numbers:
        index = int(spawn_number) - 1

        if 0 <= index < len(spawn_points):
            valid_indices.append(index)
        else:
            print(
                "WARNING: spawn {} không hợp lệ; map chỉ có {} spawn.".format(
                    spawn_number,
                    len(spawn_points),
                )
            )

    # Nếu safe list không phù hợp với map hiện tại, dùng tất cả spawn.
    if not valid_indices:
        print(
            "WARNING: không có safe spawn hợp lệ. "
            "Fallback sang toàn bộ spawn point."
        )
        valid_indices = list(range(len(spawn_points)))

    return spawn_points, valid_indices


def spawn_main_vehicle(
    world,
    blueprint,
    spawn_points,
    candidate_indices,
):
    """
    Thử spawn lần lượt trên danh sách candidate đã shuffle.
    """
    order = list(candidate_indices)
    random.shuffle(order)

    for index in order:
        original = spawn_points[index]

        transform = carla.Transform(
            carla.Location(
                x=original.location.x,
                y=original.location.y,
                z=original.location.z + 0.25,
            ),
            original.rotation,
        )

        vehicle = world.try_spawn_actor(
            blueprint,
            transform,
        )

        if vehicle is not None:
            print(
                "\nSpawn vehicle tại point {}/{}.".format(
                    index + 1,
                    len(spawn_points),
                )
            )
            return vehicle, index

    raise RuntimeError(
        "Không spawn được {} tại các spawn đã chọn.".format(
            VEHICLE_BLUEPRINT_ID
        )
    )


def destroy_sensor(camera_obj):
    if camera_obj is None:
        return

    try:
        camera_obj.sensor.stop()
    except Exception:
        pass

    try:
        camera_obj.sensor.destroy()
    except Exception:
        pass


def destroy_vehicle(vehicle):
    if vehicle is None:
        return

    try:
        vehicle.set_autopilot(False)
    except Exception:
        pass

    try:
        vehicle.destroy()
    except Exception:
        pass


def wait_for_camera(camera_obj, timeout_s=10.0):
    start = time.time()

    # RGBDatasetCamera không có front_camera buffer.
    # Chỉ cần đợi callback chạy ít nhất một frame.
    while camera_obj.frame_count == 0:
        if time.time() - start > timeout_s:
            raise TimeoutError(
                "Camera dataset không nhận frame sau {:.1f}s.".format(
                    timeout_s
                )
            )
        time.sleep(0.05)


# ================================================================
# AUTOPILOT SESSION
# ================================================================

def run_one_autopilot_session(
    client,
    world,
    traffic_manager,
    vehicle_bp,
    spawn_points,
    spawn_indices,
    args,
):
    vehicle = None
    camera_obj = None

    try:
        vehicle, spawn_index = spawn_main_vehicle(
            world,
            vehicle_bp,
            spawn_points,
            spawn_indices,
        )

        apply_automav3_wheel_physics(vehicle)
        settle_vehicle(
            world,
            vehicle,
            seconds=args.settle_seconds,
        )

        # Camera dataset dùng đúng geometry/image settings của sensors_v3.py.
        camera_obj = RGBDatasetCamera(
            vehicle,
            save_every=args.save_every,
            max_images=args.max_images,
            test_interval=args.test_interval,
        )

        # Traffic Manager settings cho main vehicle.
        traffic_manager.ignore_lights_percentage(
            vehicle,
            float(args.ignore_lights_percent),
        )
        traffic_manager.ignore_signs_percentage(
            vehicle,
            float(args.ignore_signs_percent),
        )

        # Xe CARLA scale hình học 10x, nên tốc độ kinematic mục tiêu
        # cũng scale 10x so với giới hạn vận hành xe thật.
        carla_target_speed_mps = (
            args.real_target_speed_mps
            * args.carla_geometry_scale
        )

        speed_limit_kmh, applied_speed_difference = (
            set_autopilot_target_speed(
                traffic_manager,
                vehicle,
                carla_target_speed_mps,
                fallback_percent=args.speed_difference_percent,
            )
        )

        try:
            traffic_manager.distance_to_leading_vehicle(
                vehicle,
                2.0,
            )
        except Exception:
            pass

        vehicle.set_autopilot(
            True,
            args.tm_port,
        )

        print(
            "AUTOPILOT ON | spawn={} | real_target={:.2f} m/s | "
            "CARLA_target={:.2f} m/s ({:.2f} km/h) | "
            "speed_limit={:.2f} km/h | TM difference={:.2f}%".format(
                spawn_index + 1,
                args.real_target_speed_mps,
                carla_target_speed_mps,
                carla_target_speed_mps * 3.6,
                speed_limit_kmh,
                applied_speed_difference,
            )
        )

        print(
            "STEER NOISE  | +/-{:.3f} deg | +/-{:.4f} normalized | "
            "target every {:.2f}s | smoothing={:.2f}".format(
                args.steer_noise_deg,
                steer_deg_to_normalized(args.steer_noise_deg),
                args.steer_noise_update_s,
                args.steer_noise_smoothing,
            )
        )

        print(
            "Dataset hiện có: {}/{} ảnh".format(
                count_dataset_images(),
                args.max_images,
            )
        )

        wait_for_camera(camera_obj)

        session_start = time.time()
        last_moving_time = time.time()
        last_report_time = 0.0
        previous_saved_count = camera_obj.saved_count

        # ------------------------------------------------------------
        # LAP DETECTION
        #
        # Một lap chỉ được công nhận khi:
        # 1) xe đã đi ra xa điểm xuất phát ít nhất lap_leave_radius_m;
        # 2) đã chạy tối thiểu lap_min_seconds;
        # 3) tổng quãng đường tích lũy >= lap_min_distance_m;
        # 4) quay lại trong bán kính lap_return_radius_m quanh điểm xuất phát.
        #
        # Cách này phù hợp map vòng kín và tránh trigger ngay lúc spawn.
        # ------------------------------------------------------------
        lap_start_location = vehicle.get_location()
        lap_previous_location = lap_start_location
        lap_distance_m = 0.0
        lap_armed = False
        lap_count = 0

        if args.change_lighting_every_lap:
            print(
                "LAP MODE | leave_radius={:.1f} m | return_radius={:.1f} m | "
                "min_time={:.1f}s | min_distance={:.1f} m".format(
                    args.lap_leave_radius_m,
                    args.lap_return_radius_m,
                    args.lap_min_seconds,
                    args.lap_min_distance_m,
                )
            )

        # Trạng thái steering noise hiện tại.
        target_steer_noise_deg = 0.0
        current_steer_noise_deg = 0.0
        current_steer_noise_norm = 0.0
        current_applied_steer = 0.0

        next_steer_noise_time = (
            time.time()
            + max(0.1, args.steer_noise_update_s)
        )

        while True:
            current_total = count_dataset_images()

            if current_total >= args.max_images:
                print(
                    "\nĐã đủ dataset: {}/{} ảnh.".format(
                        current_total,
                        args.max_images,
                    )
                )
                return "DONE"

            now = time.time()
            current_speed = speed_mps(vehicle)

            current_location = vehicle.get_location()
            step_distance = location_distance(
                current_location,
                lap_previous_location,
            )

            # Bỏ qua teleport/outlier bất thường khi respawn/physics giật.
            if 0.0 <= step_distance <= 10.0:
                lap_distance_m += step_distance

            lap_previous_location = current_location

            distance_to_lap_start = location_distance(
                current_location,
                lap_start_location,
            )

            if distance_to_lap_start >= args.lap_leave_radius_m:
                lap_armed = True

            if (
                args.change_lighting_every_lap
                and lap_armed
                and (now - session_start) >= args.lap_min_seconds
                and lap_distance_m >= args.lap_min_distance_m
                and distance_to_lap_start <= args.lap_return_radius_m
            ):
                lap_count += 1
                print(
                    "\nLAP COMPLETE | lap={} | time={:.1f}s | "
                    "distance={:.1f} m | dist_to_start={:.2f} m".format(
                        lap_count,
                        now - session_start,
                        lap_distance_m,
                        distance_to_lap_start,
                    )
                )
                print(
                    "-> Kết thúc vòng hiện tại để đổi sang preset ánh sáng tiếp theo."
                )
                return "LAP_COMPLETE"

            # Chọn target steering noise ngẫu nhiên mới theo chu kỳ.
            if (
                args.steer_noise_deg > 0.0
                and now >= next_steer_noise_time
            ):
                target_steer_noise_deg = random.uniform(
                    -args.steer_noise_deg,
                    args.steer_noise_deg,
                )

                next_steer_noise_time = (
                    now
                    + max(
                        0.1,
                        args.steer_noise_update_s,
                    )
                )

            if current_speed >= args.stuck_speed_mps:
                last_moving_time = now

            if now - last_moving_time > args.stuck_timeout_s:
                print(
                    "\nXe bị đứng quá {:.1f}s "
                    "(speed={:.3f} m/s) -> respawn.".format(
                        args.stuck_timeout_s,
                        current_speed,
                    )
                )
                return "RESPAWN"

            if (
                args.session_seconds > 0.0
                and now - session_start >= args.session_seconds
            ):
                print(
                    "\nHết {:.1f}s session -> đổi spawn để tăng diversity.".format(
                        args.session_seconds
                    )
                )
                return "RESPAWN"

            if now - last_report_time >= args.report_every_s:
                new_images = camera_obj.saved_count - previous_saved_count
                previous_saved_count = camera_obj.saved_count

                print(
                    "images={}/{} | speed={:.2f} m/s | "
                    "steer_noise={:+.3f} deg ({:+.4f}) | "
                    "applied_steer={:+.3f} | "
                    "session={:.1f}s | lap_dist={:.1f}m | "
                    "to_start={:.1f}m | armed={} | +{} images".format(
                        current_total,
                        args.max_images,
                        current_speed,
                        current_steer_noise_deg,
                        current_steer_noise_norm,
                        current_applied_steer,
                        now - session_start,
                        lap_distance_m,
                        distance_to_lap_start,
                        lap_armed,
                        new_images,
                    )
                )

                last_report_time = now

            try:
                world.wait_for_tick(1.0)
            except Exception:
                time.sleep(0.05)

            # Traffic Manager vừa cập nhật control cho tick mới.
            # Cộng noise sau đó để không cần vehicle_lane_offset().
            if args.steer_noise_deg > 0.0:
                alpha = clamp(
                    args.steer_noise_smoothing,
                    0.0,
                    1.0,
                )

                current_steer_noise_deg += (
                    target_steer_noise_deg
                    - current_steer_noise_deg
                ) * alpha

                (
                    current_steer_noise_norm,
                    current_applied_steer,
                ) = apply_light_steer_noise(
                    vehicle,
                    current_steer_noise_deg,
                )
            else:
                current_steer_noise_deg = 0.0
                current_steer_noise_norm = 0.0
                current_applied_steer = float(
                    vehicle.get_control().steer
                )

    finally:
        destroy_sensor(camera_obj)
        destroy_vehicle(vehicle)

        try:
            world.wait_for_tick(1.0)
        except Exception:
            time.sleep(0.1)


# ================================================================
# MAIN
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Thu RGB V3 bằng CARLA Traffic Manager autopilot "
            "với vehicle.ty.automav3."
        )
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=100,
        help="Tổng số ảnh train+test cần có.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help=(
            "Lưu 1 ảnh sau mỗi N callback camera. "
            "FAST-SAFE mặc định=1; toàn bộ timing/control loop giữ nguyên file cũ."
        ),
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=10,
        help="Mỗi N ảnh thì 1 ảnh vào test.",
    )

    parser.add_argument(
        "--real-target-speed-mps",
        type=float,
        default=DEFAULT_REAL_TARGET_SPEED_MPS,
        help=(
            "Giới hạn tốc độ tương đương trên xe thật. "
            "Mặc định 1.0 m/s."
        ),
    )
    parser.add_argument(
        "--carla-geometry-scale",
        type=float,
        default=CARLA_GEOMETRY_SCALE,
        help=(
            "Tỷ lệ geometry REAL -> CARLA. "
            "Project hiện dùng 10.0."
        ),
    )
    parser.add_argument(
        "--steer-noise-deg",
        type=float,
        default=DEFAULT_STEER_NOISE_DEG,
        help=(
            "Biên độ steering noise tối đa theo độ bánh trước. "
            "0 = tắt. Khuyên bắt đầu 0.3-0.5 deg."
        ),
    )
    parser.add_argument(
        "--steer-noise-update-s",
        type=float,
        default=DEFAULT_STEER_NOISE_UPDATE_S,
        help=(
            "Số giây giữa hai lần chọn target steering noise mới. "
            "Mặc định 2.0 s."
        ),
    )
    parser.add_argument(
        "--steer-noise-smoothing",
        type=float,
        default=DEFAULT_STEER_NOISE_SMOOTHING,
        help=(
            "Low-pass factor trong (0,1]. "
            "Nhỏ hơn = thay đổi mượt/chậm hơn. Mặc định 0.20."
        ),
    )

    parser.add_argument(
        "--speed-difference-percent",
        type=float,
        default=90.0,
        help=(
            "Fallback nếu map không trả được speed limit. "
            "Thông thường không cần chỉnh."
        ),
    )
    parser.add_argument(
        "--ignore-lights-percent",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--ignore-signs-percent",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--stuck-speed-mps",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--stuck-timeout-s",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--session-seconds",
        type=float,
        default=120.0,
        help=(
            "Sau N giây tự respawn sang vị trí khác để tăng diversity. "
            "0 = không đổi spawn theo thời gian."
        ),
    )
    parser.add_argument(
        "--change-lighting-every-lap",
        action="store_true",
        help=(
            "Giữ nguyên một preset ánh sáng cho trọn một vòng. "
            "Khi xe quay lại gần điểm xuất phát, kết thúc session và "
            "chuyển preset kế tiếp."
        ),
    )
    parser.add_argument(
        "--lap-leave-radius-m",
        type=float,
        default=8.0,
        help=(
            "Xe phải đi xa điểm xuất phát ít nhất N mét CARLA "
            "trước khi lap detector được arm. Mặc định 8.0."
        ),
    )
    parser.add_argument(
        "--lap-return-radius-m",
        type=float,
        default=3.0,
        help=(
            "Khi detector đã arm, quay lại trong bán kính N mét CARLA "
            "quanh điểm xuất phát thì có thể tính hoàn thành vòng. "
            "Mặc định 3.0."
        ),
    )
    parser.add_argument(
        "--lap-min-seconds",
        type=float,
        default=10.0,
        help=(
            "Thời gian tối thiểu trước khi được tính hoàn thành vòng. "
            "Mặc định 10 giây."
        ),
    )
    parser.add_argument(
        "--lap-min-distance-m",
        type=float,
        default=15.0,
        help=(
            "Quãng đường CARLA tích lũy tối thiểu trước khi được tính "
            "hoàn thành vòng. Mặc định 15 m."
        ),
    )

    parser.add_argument(
        "--report-every-s",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--safe-spawns",
        nargs="*",
        type=int,
        default=DEFAULT_SAFE_SPAWNS,
        help=(
            "Danh sách spawn đánh số từ 1. "
            "Ví dụ: --safe-spawns 1 4 6 7 8 9 10 11"
        ),
    )

    parser.add_argument(
        "--lighting-mode",
        choices=["cycle", "random", "fixed"],
        default="cycle",
        help=(
            "cycle = lần lượt qua preset; random = ngẫu nhiên; "
            "fixed = dùng --lighting-fixed-index."
        ),
    )
    parser.add_argument(
        "--lighting-fixed-index",
        type=int,
        default=0,
        help="Preset ánh sáng dùng khi --lighting-mode fixed.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_images <= 0:
        raise ValueError("--max-images phải > 0")

    if args.real_target_speed_mps <= 0.0:
        raise ValueError("--real-target-speed-mps phải > 0")

    if args.carla_geometry_scale <= 0.0:
        raise ValueError("--carla-geometry-scale phải > 0")

    if args.steer_noise_deg < 0.0:
        raise ValueError("--steer-noise-deg phải >= 0")

    if args.steer_noise_update_s <= 0.0:
        raise ValueError("--steer-noise-update-s phải > 0")

    if not (0.0 < args.steer_noise_smoothing <= 1.0):
        raise ValueError(
            "--steer-noise-smoothing phải nằm trong (0, 1]"
        )

    if args.lap_leave_radius_m <= 0.0:
        raise ValueError("--lap-leave-radius-m phải > 0")

    if args.lap_return_radius_m <= 0.0:
        raise ValueError("--lap-return-radius-m phải > 0")

    if args.lap_return_radius_m >= args.lap_leave_radius_m:
        raise ValueError(
            "--lap-return-radius-m phải nhỏ hơn --lap-leave-radius-m"
        )

    if args.lap_min_seconds < 0.0:
        raise ValueError("--lap-min-seconds phải >= 0")

    if args.lap_min_distance_m <= 0.0:
        raise ValueError("--lap-min-distance-m phải > 0")

    random.seed(args.seed)

    print("=" * 78)
    print("COLLECT RGB V3 - CARLA AUTOPILOT | FAST-SAFE")
    print("=" * 78)
    print("Vehicle :", VEHICLE_BLUEPRINT_ID)
    print(
        "Camera  : {}x{} @ {} FPS | FOV={}".format(
            FRONT_CAMERA_WIDTH,
            FRONT_CAMERA_HEIGHT,
            FRONT_CAMERA_FPS,
            FRONT_CAMERA_FOV,
        )
    )
    print(
        "XYZ     : ({:.6f}, {:.6f}, {:.6f})".format(
            FRONT_CAMERA_X,
            FRONT_CAMERA_Y,
            FRONT_CAMERA_Z,
        )
    )
    print(
        "PYR     : ({:.6f}, {:.6f}, {:.6f})".format(
            FRONT_CAMERA_PITCH,
            FRONT_CAMERA_YAW,
            FRONT_CAMERA_ROLL,
        )
    )
    print(
        "Target  : {} images | save every {} frames".format(
            args.max_images,
            args.save_every,
        )
    )
    carla_target_speed_mps = (
        args.real_target_speed_mps
        * args.carla_geometry_scale
    )

    print(
        "REAL    : target {:.2f} m/s ({:.2f} km/h)".format(
            args.real_target_speed_mps,
            args.real_target_speed_mps * 3.6,
        )
    )
    print(
        "SCALE   : {:.1f}x".format(
            args.carla_geometry_scale,
        )
    )
    print(
        "CARLA   : target {:.2f} m/s ({:.2f} km/h)".format(
            carla_target_speed_mps,
            carla_target_speed_mps * 3.6,
        )
    )
    print(
        "NOISE   : steer +/-{:.3f} deg -> +/-{:.4f} normalized | "
        "target every {:.2f}s | smoothing={:.2f}".format(
            args.steer_noise_deg,
            steer_deg_to_normalized(args.steer_noise_deg),
            args.steer_noise_update_s,
            args.steer_noise_smoothing,
        )
    )
    print("Dataset :", RGB_DATASET_ROOT)
    print(
        "LIGHT   : mode={} | presets={} | fixed_index={}".format(
            args.lighting_mode,
            len(LIGHTING_PRESETS),
            args.lighting_fixed_index,
        )
    )
    print(
        "LAP     : change_light_every_lap={} | leave={:.1f}m | "
        "return={:.1f}m | min_time={:.1f}s | min_dist={:.1f}m".format(
            args.change_lighting_every_lap,
            args.lap_leave_radius_m,
            args.lap_return_radius_m,
            args.lap_min_seconds,
            args.lap_min_distance_m,
        )
    )
    print("=" * 78)

    existing = count_dataset_images()

    if existing >= args.max_images:
        print(
            "Dataset đã có {}/{} ảnh. Không cần thu thêm.".format(
                existing,
                args.max_images,
            )
        )
        return

    client = carla.Client(
        args.host,
        args.port,
    )
    client.set_timeout(30.0)

    world = client.get_world()

    print("MAP     :", world.get_map().name)

    blueprint_library = world.get_blueprint_library()
    matches = blueprint_library.filter(VEHICLE_BLUEPRINT_ID)

    if not matches:
        raise RuntimeError(
            "Không tìm thấy blueprint {}".format(
                VEHICLE_BLUEPRINT_ID
            )
        )

    vehicle_bp = matches[0]

    spawn_points, spawn_indices = build_spawn_indices(
        world,
        args.safe_spawns,
    )

    print(
        "Spawns  : {}".format(
            [index + 1 for index in spawn_indices]
        )
    )

    traffic_manager = client.get_trafficmanager(
        args.tm_port
    )

    # Không bật synchronous mode ở đây vì project hiện tại
    # đang chạy world asynchronous.
    try:
        traffic_manager.set_random_device_seed(
            int(args.seed)
        )
    except Exception:
        pass

    session_index = 0
    completed_laps = 0
    lighting_cycle_index = 0

    try:
        while count_dataset_images() < args.max_images:
            session_index += 1

            print(
                "\n" + "=" * 78
            )
            print(
                "AUTOPILOT SESSION {}".format(
                    session_index
                )
            )
            print("=" * 78)

            if args.lighting_mode == "cycle":
                if args.change_lighting_every_lap:
                    lighting_index = lighting_cycle_index % len(LIGHTING_PRESETS)
                else:
                    lighting_index = (session_index - 1) % len(LIGHTING_PRESETS)
            elif args.lighting_mode == "random":
                lighting_index = random.randrange(len(LIGHTING_PRESETS))
            else:
                lighting_index = int(args.lighting_fixed_index) % len(LIGHTING_PRESETS)

            if args.change_lighting_every_lap:
                print(
                    "LAP PLAN | completed_laps={} | current_preset_index={}".format(
                        completed_laps,
                        lighting_index,
                    )
                )

            apply_lighting_preset(
                world,
                lighting_index,
            )

            # Cho shadow/weather ổn định vài tick trước khi capture.
            for _ in range(3):
                try:
                    world.wait_for_tick(1.0)
                except Exception:
                    time.sleep(0.05)

            result = run_one_autopilot_session(
                client,
                world,
                traffic_manager,
                vehicle_bp,
                spawn_points,
                spawn_indices,
                args,
            )

            if result == "DONE":
                break

            if result == "LAP_COMPLETE":
                completed_laps += 1

                if args.lighting_mode == "cycle":
                    lighting_cycle_index += 1

                print(
                    "LAP SUMMARY | completed_laps={} | "
                    "next_lighting_index={}".format(
                        completed_laps,
                        lighting_cycle_index % len(LIGHTING_PRESETS),
                    )
                )

            elif args.change_lighting_every_lap:
                print(
                    "LAP NOT COMPLETE | result={} -> giữ nguyên preset "
                    "ở lần thử tiếp theo.".format(result)
                )

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nNgười dùng dừng collector.")

    except Exception as error:
        print("\n===== COLLECTOR ERROR =====")
        print(error)
        traceback.print_exc()
        print("===========================\n")
        raise

    finally:
        final_train = count_pngs(
            RGB_DATASET_TRAIN_DIR
        )
        final_test = count_pngs(
            RGB_DATASET_TEST_DIR
        )

        print("\n" + "=" * 78)
        print("DATASET SUMMARY")
        print("=" * 78)
        print(
            "Train : {} | {}".format(
                final_train,
                RGB_DATASET_TRAIN_DIR,
            )
        )
        print(
            "Test  : {} | {}".format(
                final_test,
                RGB_DATASET_TEST_DIR,
            )
        )
        print(
            "Total : {}/{}".format(
                final_train + final_test,
                args.max_images,
            )
        )
        print("=" * 78)


if __name__ == "__main__":
    main()



# ================================================================
# GỢI Ý CHẠY
#
# Smoke:
# python .\collect_rgb_v3_multilight.py --max-images 300 --save-every 10 --real-target-speed-mps 0.4 --session-seconds 30 --lighting-mode cycle
#
# Thu chính, tiếp tục trên dataset_v4 hiện có:
# python .\collect_rgb_v3_multilight.py --max-images 30000 --save-every 10 --real-target-speed-mps 0.4 --session-seconds 45 --steer-noise-deg 0.7 --lighting-mode cycle
#
# Test riêng một điều kiện ánh sáng:
# python .\collect_rgb_v3_multilight.py --max-images 1000 --save-every 10 --real-target-speed-mps 0.4 --session-seconds 0 --steer-noise-deg 0.0 --lighting-mode fixed --lighting-fixed-index 0


# ================================================================
# LAP MODE EXAMPLE
#
# Mỗi vòng kín dùng đúng 1 preset ánh sáng; hoàn thành vòng mới đổi preset:
#
# python .\collect_rgb_v3_multilight_lap.py `
#   --max-images 30000 `
#   --save-every 10 `
#   --real-target-speed-mps 0.20 `
#   --session-seconds 0 `
#   --steer-noise-deg 0 `
#   --lighting-mode cycle `
#   --change-lighting-every-lap
#
# Nếu detector chưa nhận vòng:
# - xem log lap_dist / to_start / armed
# - giảm --lap-min-distance-m nếu map rất nhỏ
# - tăng --lap-return-radius-m nếu xe không đi sát đúng điểm spawn khi quay về
# ================================================================
