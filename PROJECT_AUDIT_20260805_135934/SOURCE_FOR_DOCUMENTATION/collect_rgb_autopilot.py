import os
import time
import weakref
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from simulation.connection import carla
from simulation.sensors import (
    FRONT_CAMERA_WIDTH,
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_FPS,
    FRONT_CAMERA_FOV,
    FRONT_CAMERA_X,
    FRONT_CAMERA_Y,
    FRONT_CAMERA_YAW,
    FRONT_CAMERA_ROLL,
    FRONT_CAMERA_Z,
    FRONT_CAMERA_PITCH,
)


COLLECT_CAMERA_Z = FRONT_CAMERA_Z
COLLECT_CAMERA_PITCH = FRONT_CAMERA_PITCH
HOST = "127.0.0.1"
PORT = 2000
TRAFFIC_MANAGER_PORT = 8000
EXPECTED_MAP_KEYWORD = "mapden"

SAVE_ROOT = os.path.join("autoencoder", "dataset_rgb_autopilot")
TRAIN_DIR = os.path.join(SAVE_ROOT, "train", "rgb")
TEST_DIR = os.path.join(SAVE_ROOT, "test", "rgb")

MAX_IMAGES = 16000
IMAGES_PER_SPAWN = 2000
SAVE_EVERY = 10
TEST_INTERVAL = 10
SAFE_SPAWN_NUMBERS = [1, 4, 6, 7, 8, 9, 10, 11]
VEHICLE_FILTER = "vehicle.tesla.model3"

RGB_GAMMA = 2.2
RGB_EXPOSURE_COMPENSATION = 0.0

SPAWN_HEIGHT_OFFSET = 0.20
SETTLE_SECONDS = 1.5
STUCK_TIMEOUT_SECONDS = 12.0
MIN_MOVING_SPEED_KMH = 1.0
PROGRESS_PRINT_INTERVAL_SECONDS = 2.0
RANDOMIZE_WEATHER = True

WEATHER_PRESETS = [
    ("ClearNoon", carla.WeatherParameters.ClearNoon),
    ("CloudyNoon", carla.WeatherParameters.CloudyNoon),
    ("WetNoon", carla.WeatherParameters.WetNoon),
    ("ClearSunset", carla.WeatherParameters.ClearSunset),
]


def count_png_files(directory):
    if not os.path.isdir(directory):
        return 0
    return len([name for name in os.listdir(directory) if name.lower().endswith(".png")])


def ensure_dataset_directories():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


def get_existing_image_count():
    return count_png_files(TRAIN_DIR) + count_png_files(TEST_DIR)


def get_next_filename_index():
    indices = []
    for directory in (TRAIN_DIR, TEST_DIR):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.lower().endswith(".png"):
                continue
            try:
                indices.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(indices) + 1 if indices else 0


def clone_transform(transform):
    return carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
        ),
        carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll,
        ),
    )


def get_vehicle_speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5 * 3.6


class AutopilotRGBCamera:
    def __init__(self, vehicle, spawn_number, start_index, max_total_images, max_spawn_images):
        self.vehicle = vehicle
        self.spawn_number = int(spawn_number)
        self.frame_count = 0
        self.saved_count = int(start_index)
        self.session_saved = 0
        self.max_total_images = int(max_total_images)
        self.max_spawn_images = int(max_spawn_images)

        world = vehicle.get_world()
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(FRONT_CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(FRONT_CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(FRONT_CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(1.0 / FRONT_CAMERA_FPS))

        if camera_bp.has_attribute("enable_postprocess_effects"):
            camera_bp.set_attribute("enable_postprocess_effects", "true")
        if camera_bp.has_attribute("gamma"):
            camera_bp.set_attribute("gamma", str(RGB_GAMMA))
        if camera_bp.has_attribute("exposure_mode"):
            camera_bp.set_attribute("exposure_mode", "histogram")
        if camera_bp.has_attribute("exposure_compensation"):
            camera_bp.set_attribute("exposure_compensation", str(RGB_EXPOSURE_COMPENSATION))

        for attribute, value in (
            ("bloom_intensity", "0.0"),
            ("lens_flare_intensity", "0.0"),
            ("motion_blur_intensity", "0.0"),
            ("chromatic_aberration_intensity", "0.0"),
        ):
            if camera_bp.has_attribute(attribute):
                camera_bp.set_attribute(attribute, value)

        camera_transform = carla.Transform(
            carla.Location(x=FRONT_CAMERA_X, y=FRONT_CAMERA_Y, z=COLLECT_CAMERA_Z),
            carla.Rotation(
                pitch=COLLECT_CAMERA_PITCH,
                yaw=FRONT_CAMERA_YAW,
                roll=FRONT_CAMERA_ROLL,
            ),
        )

        self.sensor = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid,
        )

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: AutopilotRGBCamera._save_frame(weak_self, image))

    @property
    def reached_total_limit(self):
        return self.saved_count >= self.max_total_images

    @property
    def reached_spawn_limit(self):
        return self.session_saved >= self.max_spawn_images

    @staticmethod
    def _save_frame(weak_self, image):
        self = weak_self()
        if self is None or self.reached_total_limit or self.reached_spawn_limit:
            return

        self.frame_count += 1
        if self.frame_count % SAVE_EVERY != 0:
            return

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]
        rgb = bgr[:, :, ::-1].copy()

        filename = "rgb_{:06d}_spawn_{:02d}_carla_{:08d}.png".format(
            self.saved_count,
            self.spawn_number,
            image.frame,
        )

        output_dir = TEST_DIR if self.saved_count % TEST_INTERVAL == 0 else TRAIN_DIR
        output_path = os.path.join(output_dir, filename)

        try:
            Image.fromarray(rgb).save(output_path)
        except OSError as error:
            print("\n[AUTOPILOT RGB] Không thể lưu:", output_path, error)
            return

        self.saved_count += 1
        self.session_saved += 1

        if self.saved_count % 100 == 0:
            print(
                "\n[AUTOPILOT RGB] {}/{} ảnh | spawn {} | spawn progress {}/{}".format(
                    self.saved_count,
                    self.max_total_images,
                    self.spawn_number,
                    self.session_saved,
                    self.max_spawn_images,
                )
            )

    def destroy(self):
        try:
            if self.sensor is not None:
                self.sensor.stop()
        except Exception:
            pass
        try:
            if self.sensor is not None and self.sensor.is_alive:
                self.sensor.destroy()
        except Exception:
            pass


def choose_weather(world):
    if not RANDOMIZE_WEATHER:
        return "CurrentWeather"
    weather_name, weather = random.choice(WEATHER_PRESETS)
    world.set_weather(weather)
    return weather_name


def spawn_vehicle(world, spawn_points, spawn_number):
    spawn_index = spawn_number - 1
    if spawn_index < 0 or spawn_index >= len(spawn_points):
        raise RuntimeError(
            "Spawn {} không hợp lệ. Map có {} spawn.".format(spawn_number, len(spawn_points))
        )

    candidates = world.get_blueprint_library().filter(VEHICLE_FILTER)
    if not candidates:
        raise RuntimeError("Không tìm thấy vehicle blueprint: " + VEHICLE_FILTER)

    vehicle_bp = random.choice(candidates)
    spawn_transform = clone_transform(spawn_points[spawn_index])
    spawn_transform.location.z += SPAWN_HEIGHT_OFFSET

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if vehicle is None:
        raise RuntimeError("Không spawn được xe tại spawn {}.".format(spawn_number))

    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    time.sleep(SETTLE_SECONDS)
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
    return vehicle


def destroy_vehicle(vehicle):
    try:
        if vehicle is not None and vehicle.is_alive:
            vehicle.set_autopilot(False)
    except Exception:
        pass
    try:
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
    except Exception:
        pass


def run_spawn_session(world, spawn_points, spawn_number, start_index):
    weather_name = choose_weather(world)
    vehicle = None
    camera = None

    try:
        vehicle = spawn_vehicle(world, spawn_points, spawn_number)
        vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)

        camera = AutopilotRGBCamera(
            vehicle=vehicle,
            spawn_number=spawn_number,
            start_index=start_index,
            max_total_images=MAX_IMAGES,
            max_spawn_images=IMAGES_PER_SPAWN,
        )

        print("\n" + "=" * 62)
        print("Spawn               :", spawn_number)
        print("Weather             :", weather_name)
        print("Total dataset       : {}/{}".format(start_index, MAX_IMAGES))
        print("Images this spawn   : 0/{}".format(IMAGES_PER_SPAWN))
        print("Camera Z            : {:.3f} m".format(COLLECT_CAMERA_Z))
        print("Camera pitch        : {:.1f} deg".format(COLLECT_CAMERA_PITCH))
        print("Save every          : {} frames".format(SAVE_EVERY))
        print("=" * 62)

        last_progress_time = time.time()
        last_moving_time = time.time()

        while not camera.reached_total_limit and not camera.reached_spawn_limit:
            world.wait_for_tick()
            speed_kmh = get_vehicle_speed_kmh(vehicle)

            if speed_kmh >= MIN_MOVING_SPEED_KMH:
                last_moving_time = time.time()

            if time.time() - last_moving_time > STUCK_TIMEOUT_SECONDS:
                print(
                    "\n[AUTOPILOT RGB] Xe bị kẹt tại spawn {}. Chuyển spawn.".format(
                        spawn_number
                    )
                )
                break

            if time.time() - last_progress_time >= PROGRESS_PRINT_INTERVAL_SECONDS:
                print(
                    "\rSpawn {:02d} | saved {}/{} | this spawn {}/{} | speed {:.1f} km/h".format(
                        spawn_number,
                        camera.saved_count,
                        MAX_IMAGES,
                        camera.session_saved,
                        IMAGES_PER_SPAWN,
                        speed_kmh,
                    ),
                    end="",
                )
                last_progress_time = time.time()

        print()
        return camera.saved_count

    finally:
        if camera is not None:
            camera.destroy()
        destroy_vehicle(vehicle)
        try:
            world.wait_for_tick(2.0)
        except Exception:
            time.sleep(0.2)


def main():
    ensure_dataset_directories()

    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = client.get_world()
    carla_map = world.get_map()

    print("Current map:", carla_map.name)

    if EXPECTED_MAP_KEYWORD and EXPECTED_MAP_KEYWORD.lower() not in carla_map.name.lower():
        raise RuntimeError(
            "Map hiện tại không phải '{}': {}".format(EXPECTED_MAP_KEYWORD, carla_map.name)
        )

    traffic_manager = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
    traffic_manager.set_synchronous_mode(False)

    spawn_points = carla_map.get_spawn_points()
    valid_spawn_numbers = [
        number for number in SAFE_SPAWN_NUMBERS if 1 <= number <= len(spawn_points)
    ]

    if not valid_spawn_numbers:
        raise RuntimeError("Không có safe spawn hợp lệ.")

    total_saved = max(get_existing_image_count(), get_next_filename_index())
    print("Existing dataset:", total_saved, "images")

    if total_saved >= MAX_IMAGES:
        print("Dataset đã đủ {} ảnh.".format(MAX_IMAGES))
        return

    spawn_order = list(valid_spawn_numbers)

    try:
        while total_saved < MAX_IMAGES:
            random.shuffle(spawn_order)
            for spawn_number in spawn_order:
                if total_saved >= MAX_IMAGES:
                    break
                try:
                    total_saved = run_spawn_session(
                        world=world,
                        spawn_points=spawn_points,
                        spawn_number=spawn_number,
                        start_index=total_saved,
                    )
                except RuntimeError as error:
                    print(
                        "\n[AUTOPILOT RGB] Bỏ qua spawn {}: {}".format(
                            spawn_number,
                            error,
                        )
                    )

        print("\nĐã thu đủ {} ảnh.".format(total_saved))

    except KeyboardInterrupt:
        print("\nStopped by user. Dataset đã được giữ lại.")


if __name__ == "__main__":
    main()