import math
import os
import weakref

import numpy as np
import pygame
from PIL import Image

from simulation.connection import carla
from simulation.settings import RGB_CAMERA, SSC_CAMERA


# ================================================================
# THÔNG SỐ CAMERA PHÍA TRƯỚC
# Đo trên xe F1/10 thật và scale ×10 cho xe mô phỏng.
# ================================================================

FRONT_CAMERA_WIDTH = 160
FRONT_CAMERA_HEIGHT = 80
FRONT_CAMERA_FPS = 30
FRONT_CAMERA_FOV = 125.0

FRONT_CAMERA_X = 1.898415
FRONT_CAMERA_Y = 0.0
# FRONT_CAMERA_Z = 1.357832
FRONT_CAMERA_Z = 1.45

# FRONT_CAMERA_PITCH = -14.9
FRONT_CAMERA_PITCH = -16.5
FRONT_CAMERA_YAW = 0.0
FRONT_CAMERA_ROLL = 0.0


# ================================================================
# CẤU HÌNH THU DATASET RGB
# ================================================================

RGB_DATASET_ROOT = os.path.join("autoencoder", "dataset_rgb")
RGB_DATASET_TRAIN_DIR = os.path.join(RGB_DATASET_ROOT, "train", "rgb")
RGB_DATASET_TEST_DIR = os.path.join(RGB_DATASET_ROOT, "test", "rgb")

RGB_DATASET_SAVE_EVERY = 5
RGB_DATASET_MAX_IMAGES = 100
RGB_DATASET_TEST_INTERVAL = 10

# RGB camera configuration for VAE dataset collection.
# Keep 160x80 so the VAE output can later replace the semantic encoder input
# without changing the PPO observation shape.
RGB_GAMMA = 2.2
RGB_EXPOSURE_COMPENSATION = 0.0


# ================================================================
# CAMERA SEMANTIC
# Giữ nguyên tên CameraSensor để PPO/checkpoint cũ vẫn hoạt động.
# ================================================================

class CameraSensor:

    def __init__(self, vehicle):
        self.sensor_name = SSC_CAMERA
        self.parent = vehicle
        self.front_camera = []

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(
                weak_self, image
            )
        )

    def _set_camera_sensor(self, world):
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        camera_bp.set_attribute("image_size_x", str(FRONT_CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(FRONT_CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(FRONT_CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(1.0 / FRONT_CAMERA_FPS))

        camera_transform = carla.Transform(
            carla.Location(
                x=FRONT_CAMERA_X,
                y=FRONT_CAMERA_Y,
                z=FRONT_CAMERA_Z,
            ),
            carla.Rotation(
                pitch=FRONT_CAMERA_PITCH,
                yaw=FRONT_CAMERA_YAW,
                roll=FRONT_CAMERA_ROLL,
            ),
        )

        return world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if self is None:
            return

        image.convert(carla.ColorConverter.CityScapesPalette)

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        semantic_image = array[:, :, :3].copy()

        # Chỉ giữ frame mới nhất để tránh tăng RAM liên tục.
        self.front_camera.clear()
        self.front_camera.append(semantic_image)


# ================================================================
# CAMERA RGB THU DATASET
# Không tham gia điều khiển PPO semantic hiện tại.
# ================================================================

class RGBDatasetCamera:

    def __init__(
        self,
        vehicle,
        save_every=RGB_DATASET_SAVE_EVERY,
        max_images=RGB_DATASET_MAX_IMAGES,
        test_interval=RGB_DATASET_TEST_INTERVAL,
    ):
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle

        self.save_every = max(1, int(save_every))
        self.max_images = max(1, int(max_images))
        self.test_interval = max(2, int(test_interval))

        self.frame_count = 0

        os.makedirs(RGB_DATASET_TRAIN_DIR, exist_ok=True)
        os.makedirs(RGB_DATASET_TEST_DIR, exist_ok=True)

        # The RGB camera is recreated after every episode reset.
        # Count existing images so files are not restarted from zero and the
        # global dataset stops at exactly max_images.
        existing_train = len([
            name for name in os.listdir(RGB_DATASET_TRAIN_DIR)
            if name.lower().endswith(".png")
        ])
        existing_test = len([
            name for name in os.listdir(RGB_DATASET_TEST_DIR)
            if name.lower().endswith(".png")
        ])
        self.saved_count = existing_train + existing_test

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: RGBDatasetCamera._save_rgb_frame(
                weak_self, image
            )
        )

    def _set_camera_sensor(self, world):
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        camera_bp.set_attribute("image_size_x", str(FRONT_CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(FRONT_CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(FRONT_CAMERA_FOV))
        camera_bp.set_attribute("sensor_tick", str(1.0 / FRONT_CAMERA_FPS))

        # Enable CARLA's tonemapping and automatic exposure. Disabling the
        # complete post-process pipeline caused the road/grass to become nearly
        # black while the sky was clipped to white on this custom map.
        if camera_bp.has_attribute("enable_postprocess_effects"):
            camera_bp.set_attribute("enable_postprocess_effects", "true")

        if camera_bp.has_attribute("gamma"):
            camera_bp.set_attribute("gamma", str(RGB_GAMMA))

        if camera_bp.has_attribute("exposure_mode"):
            camera_bp.set_attribute("exposure_mode", "histogram")

        if camera_bp.has_attribute("exposure_compensation"):
            camera_bp.set_attribute(
                "exposure_compensation",
                str(RGB_EXPOSURE_COMPENSATION),
            )

        # Keep useful exposure/tonemapping but remove cinematic artifacts that
        # are undesirable in a learning dataset.
        for attribute, value in (
            ("bloom_intensity", "0.0"),
            ("lens_flare_intensity", "0.0"),
            ("motion_blur_intensity", "0.0"),
            ("chromatic_aberration_intensity", "0.0"),
        ):
            if camera_bp.has_attribute(attribute):
                camera_bp.set_attribute(attribute, value)

        camera_transform = carla.Transform(
            carla.Location(
                x=FRONT_CAMERA_X,
                y=FRONT_CAMERA_Y,
                z=FRONT_CAMERA_Z,
            ),
            carla.Rotation(
                pitch=FRONT_CAMERA_PITCH,
                yaw=FRONT_CAMERA_YAW,
                roll=FRONT_CAMERA_ROLL,
            ),
        )

        return world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _save_rgb_frame(weak_self, image):
        self = weak_self()
        if self is None or self.saved_count >= self.max_images:
            return

        self.frame_count += 1
        if self.frame_count % self.save_every != 0:
            return

        image.convert(carla.ColorConverter.Raw)

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        bgr_image = array[:, :, :3]
        rgb_image = bgr_image[:, :, ::-1].copy()

        filename = (
            f"rgb_{self.saved_count:06d}_"
            f"carla_{image.frame:08d}.png"
        )

        if self.saved_count % self.test_interval == 0:
            output_path = os.path.join(RGB_DATASET_TEST_DIR, filename)
        else:
            output_path = os.path.join(RGB_DATASET_TRAIN_DIR, filename)

        try:
            Image.fromarray(rgb_image).save(output_path)
        except OSError as error:
            print("[RGB DATASET] Không thể lưu ảnh:", output_path, error)
            return

        self.saved_count += 1

        if self.saved_count % 100 == 0:
            print(
                f"[RGB DATASET] Đã lưu "
                f"{self.saved_count}/{self.max_images} ảnh"
            )

        if self.saved_count == self.max_images:
            print(
                f"[RGB DATASET] Đã thu đủ "
                f"{self.max_images} ảnh."
            )


# ================================================================
# CAMERA QUAN SÁT MÔI TRƯỜNG
# ================================================================

class CameraSensorEnv:

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode(
            (720, 720), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensorEnv._get_third_person_camera(
                weak_self, image
            )
        )

    def _set_camera_sensor(self, world):
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        camera_bp.set_attribute("image_size_x", "720")
        camera_bp.set_attribute("image_size_y", "720")
        camera_bp.set_attribute("sensor_tick", str(1.0 / 30.0))

        camera_transform = carla.Transform(
            carla.Location(x=-4.0, y=0.0, z=2.0),
            carla.Rotation(pitch=-12.0, yaw=0.0, roll=0.0),
        )

        return world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.parent,
        )

    @staticmethod
    def _get_third_person_camera(weak_self, image):
        self = weak_self()
        if self is None:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        bgr_image = array[:, :, :3]
        rgb_image = bgr_image[:, :, ::-1].copy()

        self.surface = pygame.surfarray.make_surface(
            rgb_image.swapaxes(0, 1)
        )
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()


# ================================================================
# COLLISION SENSOR
# ================================================================

class CollisionSensor:

    def __init__(self, vehicle):
        self.sensor_name = "sensor.other.collision"
        self.parent = vehicle
        self.collision_data = []

        world = self.parent.get_world()
        self.sensor = self._set_collision_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(
                weak_self, event
            )
        )

    def _set_collision_sensor(self, world):
        collision_sensor_bp = world.get_blueprint_library().find(
            self.sensor_name
        )

        sensor_relative_transform = carla.Transform(
            carla.Location(x=1.3, y=0.0, z=0.5)
        )

        return world.spawn_actor(
            collision_sensor_bp,
            sensor_relative_transform,
            attach_to=self.parent,
        )

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if self is None:
            return

        impulse = event.normal_impulse
        intensity = math.sqrt(
            impulse.x ** 2
            + impulse.y ** 2
            + impulse.z ** 2
        )
        self.collision_data.append(intensity)
