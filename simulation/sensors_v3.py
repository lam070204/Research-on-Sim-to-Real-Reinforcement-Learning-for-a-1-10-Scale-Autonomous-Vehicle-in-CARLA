import math
import os
import weakref

import numpy as np
import pygame
from PIL import Image

from simulation.connection import carla
from simulation.settings import RGB_CAMERA


# ================================================================
# SENSORS V3 - SIM-TO-REAL
#
# Thông số camera cập nhật theo số đo xe thật 1/10.
# ================================================================

FRONT_CAMERA_WIDTH = 160
FRONT_CAMERA_HEIGHT = 80
FRONT_CAMERA_FPS = 30
FRONT_CAMERA_FOV = 125.0

# Số đo xe thật
REAL_WHEELBASE_M = 0.258
REAL_WHEEL_DIAMETER_M = 0.067
REAL_WHEEL_RADIUS_M = REAL_WHEEL_DIAMETER_M / 2.0

REAL_CAMERA_FROM_REAR_AXLE_M = 0.243
REAL_CAMERA_LATERAL_M = 0.0
REAL_CAMERA_HEIGHT_M = 0.180
REAL_CAMERA_GROUND_DISTANCE_M = 0.506

# Tỷ lệ model đang dùng trong CARLA
CARLA_SCALE = 10.0

# Mốc local của model Blender/CARLA hiện tại
MODEL_REAR_AXLE_X = -1.318814
MODEL_GROUND_Z = -0.008972

# Camera local trong CARLA
FRONT_CAMERA_X = (
    MODEL_REAR_AXLE_X
    + REAL_CAMERA_FROM_REAR_AXLE_M * CARLA_SCALE
)
FRONT_CAMERA_Y = REAL_CAMERA_LATERAL_M * CARLA_SCALE
FRONT_CAMERA_Z = (
    MODEL_GROUND_Z
    + REAL_CAMERA_HEIGHT_M * CARLA_SCALE
)

# Tính trực tiếp từ số đo xe thật
FRONT_CAMERA_PITCH = -math.degrees(
    math.atan2(
        REAL_CAMERA_HEIGHT_M,
        REAL_CAMERA_GROUND_DISTANCE_M,
    )
)
FRONT_CAMERA_YAW = 0.0
FRONT_CAMERA_ROLL = 0.0


# ================================================================
# RGB IMAGE SETTINGS
# Camera PPO và camera thu dataset dùng cùng thiết lập ảnh.
# ================================================================

RGB_GAMMA = 2.2
RGB_EXPOSURE_COMPENSATION = 0.0


# ================================================================
# DATASET RGB V3
# ================================================================

RGB_DATASET_ROOT = os.path.join("datasets", "rgb_v3")
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

# Mặc định nhỏ để test collector; khi thu chính thức driver có thể truyền khác.
RGB_DATASET_SAVE_EVERY = 5
RGB_DATASET_MAX_IMAGES = 100
RGB_DATASET_TEST_INTERVAL = 10


# ================================================================
# HÀM DÙNG CHUNG CHO CAMERA RGB PHÍA TRƯỚC
# ================================================================

def _configure_front_rgb_blueprint(camera_bp):
    """Cấu hình RGB giống nhau cho PPO runtime và dataset VAE."""

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
        camera_bp.set_attribute(
            "exposure_compensation",
            str(RGB_EXPOSURE_COMPENSATION),
        )

    for attribute, value in (
        ("bloom_intensity", "0.0"),
        ("lens_flare_intensity", "0.0"),
        ("motion_blur_intensity", "0.0"),
        ("chromatic_aberration_intensity", "0.0"),
    ):
        if camera_bp.has_attribute(attribute):
            camera_bp.set_attribute(attribute, value)


def _get_front_camera_transform():
    return carla.Transform(
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


def _carla_bgra_to_rgb(image):
    """CARLA raw BGRA -> numpy RGB uint8."""

    image.convert(carla.ColorConverter.Raw)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    bgr_image = array[:, :, :3]
    rgb_image = bgr_image[:, :, ::-1].copy()

    return rgb_image


# ================================================================
# CAMERA RGB CHÍNH CHO VAE + PPO V3
# ================================================================

class CameraSensor:
    """
    Camera phía trước dùng trực tiếp cho observation RGB V3.

    self.front_camera[-1]:
        numpy.ndarray, shape (80, 160, 3), RGB uint8
    """

    def __init__(self, vehicle):
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.front_camera = []

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensor._get_front_camera_data(
                weak_self,
                image,
            )
        )

    def _set_camera_sensor(self, world):
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        _configure_front_rgb_blueprint(camera_bp)

        return world.spawn_actor(
            camera_bp,
            _get_front_camera_transform(),
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _get_front_camera_data(weak_self, image):
        self = weak_self()
        if self is None:
            return

        rgb_image = _carla_bgra_to_rgb(image)

        # Chỉ giữ frame mới nhất để không tăng RAM liên tục.
        self.front_camera.clear()
        self.front_camera.append(rgb_image)


# ================================================================
# CAMERA RGB THU DATASET VAE V3
# Geometry + image settings giống hệt CameraSensor.
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

        existing_train = len([
            name
            for name in os.listdir(RGB_DATASET_TRAIN_DIR)
            if name.lower().endswith(".png")
        ])
        existing_test = len([
            name
            for name in os.listdir(RGB_DATASET_TEST_DIR)
            if name.lower().endswith(".png")
        ])

        self.saved_count = existing_train + existing_test

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: RGBDatasetCamera._save_rgb_frame(
                weak_self,
                image,
            )
        )

    def _set_camera_sensor(self, world):
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        _configure_front_rgb_blueprint(camera_bp)

        return world.spawn_actor(
            camera_bp,
            _get_front_camera_transform(),
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

        rgb_image = _carla_bgra_to_rgb(image)

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
# Chỉ để hiển thị, KHÔNG đưa vào VAE/PPO.
# ================================================================

class CameraSensorEnv:

    def __init__(self, vehicle):
        pygame.init()
        self.display = pygame.display.set_mode(
            (720, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.surface = None

        world = self.parent.get_world()
        self.sensor = self._set_camera_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda image: CameraSensorEnv._get_third_person_camera(
                weak_self,
                image,
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

        rgb_image = _carla_bgra_to_rgb(image)

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
                weak_self,
                event,
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