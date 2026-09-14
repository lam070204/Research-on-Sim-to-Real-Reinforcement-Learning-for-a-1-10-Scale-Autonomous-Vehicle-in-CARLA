import math
import os
import weakref

import numpy as np
import pygame
from PIL import Image

from simulation.carla_connection_v5 import carla
from simulation.simulation_settings_v5 import RGB_CAMERA
from vehicle_specs_v5 import (
    CARLA_GEOMETRY_SCALE,
    FRONT_CAMERA_WIDTH,
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_FPS,
    FRONT_CAMERA_FOV_DEG,
    FRONT_CAMERA_X,
    FRONT_CAMERA_Y,
    FRONT_CAMERA_Z,
    FRONT_CAMERA_PITCH_DEG,
    FRONT_CAMERA_YAW_DEG,
    FRONT_CAMERA_ROLL_DEG,
    IMU_X,
    IMU_Y,
    IMU_Z,
    IMU_PITCH_DEG,
    IMU_YAW_DEG,
    IMU_ROLL_DEG,
    IMU_FPS,
)


# ================================================================
# SENSORS V3 - SIM-TO-REAL
#
# Geometry/camera/IMU mount được lấy từ vehicle_specs_v5.py
# để chỉ có MỘT nguồn thông số chuẩn trong project.
# ================================================================

# Alias để giữ tương thích với code cũ trong file này.
FRONT_CAMERA_FOV = FRONT_CAMERA_FOV_DEG
FRONT_CAMERA_PITCH = FRONT_CAMERA_PITCH_DEG
FRONT_CAMERA_YAW = FRONT_CAMERA_YAW_DEG
FRONT_CAMERA_ROLL = FRONT_CAMERA_ROLL_DEG

CARLA_SCALE = CARLA_GEOMETRY_SCALE


# ================================================================
# RGB IMAGE SETTINGS
# Camera PPO vÃ  camera thu dataset dÃ¹ng cÃ¹ng thiáº¿t láº­p áº£nh.
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

# Máº·c Ä‘á»‹nh nhá» Ä‘á»ƒ test collector; khi thu chÃ­nh thá»©c driver cÃ³ thá»ƒ truyá»n khÃ¡c.
RGB_DATASET_SAVE_EVERY = 5
RGB_DATASET_MAX_IMAGES = 100
RGB_DATASET_TEST_INTERVAL = 10


# ================================================================
# HÃ€M DÃ™NG CHUNG CHO CAMERA RGB PHÃA TRÆ¯á»šC
# ================================================================

def _configure_front_rgb_blueprint(camera_bp):
    """Cáº¥u hÃ¬nh RGB giá»‘ng nhau cho PPO runtime vÃ  dataset VAE."""

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
# CAMERA RGB CHÃNH CHO VAE + PPO V3
# ================================================================

class CameraSensor:
    """
    Camera phÃ­a trÆ°á»›c dÃ¹ng trá»±c tiáº¿p cho observation RGB V3.

    self.front_camera[-1]:
        numpy.ndarray, shape (80, 160, 3), RGB uint8
    """

    def __init__(self, vehicle):
        self.sensor_name = RGB_CAMERA
        self.parent = vehicle
        self.front_camera = []
        self.frame = -1
        self.timestamp = 0.0
        self.ready = False
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


        # PPO camera giữ đúng camera thật 30 FPS; control loop chạy 50 Hz.
        camera_bp.set_attribute("sensor_tick", str(1.0 / FRONT_CAMERA_FPS))
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



        self.frame = int(image.frame)

        self.timestamp = float(image.timestamp)

        self.ready = True
        # Chá»‰ giá»¯ frame má»›i nháº¥t Ä‘á»ƒ khÃ´ng tÄƒng RAM liÃªn tá»¥c.
        self.front_camera.clear()
        self.front_camera.append(rgb_image)


# ================================================================
# CAMERA RGB THU DATASET VAE V3
# Geometry + image settings giá»‘ng há»‡t CameraSensor.
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
            print("[RGB DATASET] KhÃ´ng thá»ƒ lÆ°u áº£nh:", output_path, error)
            return

        self.saved_count += 1

        if self.saved_count % 100 == 0:
            print(
                f"[RGB DATASET] ÄÃ£ lÆ°u "
                f"{self.saved_count}/{self.max_images} áº£nh"
            )

        if self.saved_count == self.max_images:
            print(
                f"[RGB DATASET] ÄÃ£ thu Ä‘á»§ "
                f"{self.max_images} áº£nh."
            )


# ================================================================
# CAMERA QUAN SÃT MÃ”I TRÆ¯á»œNG
# Chá»‰ Ä‘á»ƒ hiá»ƒn thá»‹, KHÃ”NG Ä‘Æ°a vÃ o VAE/PPO.
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
        camera_bp.set_attribute("sensor_tick", str(1.0 / FRONT_CAMERA_FPS))

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
# IMU / PROPRIOCEPTION V3 - SIM-TO-REAL
# ================================================================

IMU_SENSOR = "sensor.other.imu"
IMU_FPS = 50.0
IMU_SENSOR_TICK = 1.0 / IMU_FPS

# IMU_X / IMU_Y / IMU_Z và orientation được import trực tiếp
# từ vehicle_specs_v5.py. Không khai báo lại ở đây.
#
# Giá trị hiện tại (nếu vehicle_specs_v5.py đã được cập nhật):
#   REAL mount:  X=0.129 m từ rear axle, Y=0, Z=0.070 m từ ground
#   CARLA local: X=-0.118498, Y=0.0, Z=0.572581


IMU_NOISE_ACCEL_STDDEV = 0.0
IMU_NOISE_GYRO_STDDEV = 0.0
IMU_NOISE_GYRO_BIAS = 0.0


class IMUSensorV5:
    # SIM mapping:
    # speed     = |vehicle velocity| / CARLA_SCALE
    # yaw_rate  = sensor.other.imu.gyroscope.z [rad/s]
    # accel_x   = vehicle.get_acceleration() projected to forward / CARLA_SCALE
    #
    # Raw accelerometer cá»§a sensor.other.imu chá»‰ giá»¯ Ä‘á»ƒ debug,
    # KHÃ”NG dÃ¹ng cho PPO vÃ¬ custom vehicle hiá»‡n cÃ³ spike lá»›n.

    def __init__(self, vehicle):
        self.sensor_name = IMU_SENSOR
        self.parent = vehicle

        self.yaw_rate_rad_s = 0.0

        self.raw_imu_accel_x_mps2 = 0.0
        self.raw_imu_accel_y_mps2 = 0.0
        self.raw_imu_accel_z_mps2 = 0.0

        self.timestamp = 0.0
        self.frame = -1
        self.ready = False

        world = self.parent.get_world()
        self.sensor = self._set_imu_sensor(world)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda data: IMUSensorV5._on_imu(weak_self, data)
        )

    def _set_imu_sensor(self, world):
        imu_bp = world.get_blueprint_library().find(self.sensor_name)

        # IMU synchronous with the 50 Hz CARLA world.
        # sensor_tick=0.0 = emit every simulation tick.
        # World tick is 0.020 s, therefore effective IMU rate is exactly 50 Hz.
        # This avoids the one-frame phase lag seen with sensor_tick=0.020
        # on the current CARLA 0.9.13-dirty build.
        imu_bp.set_attribute(
            "sensor_tick",
            "0.0",
        )

        for attribute, value in (
            ("noise_accel_stddev_x", IMU_NOISE_ACCEL_STDDEV),
            ("noise_accel_stddev_y", IMU_NOISE_ACCEL_STDDEV),
            ("noise_accel_stddev_z", IMU_NOISE_ACCEL_STDDEV),
            ("noise_gyro_stddev_x", IMU_NOISE_GYRO_STDDEV),
            ("noise_gyro_stddev_y", IMU_NOISE_GYRO_STDDEV),
            ("noise_gyro_stddev_z", IMU_NOISE_GYRO_STDDEV),
            ("noise_gyro_bias_x", IMU_NOISE_GYRO_BIAS),
            ("noise_gyro_bias_y", IMU_NOISE_GYRO_BIAS),
            ("noise_gyro_bias_z", IMU_NOISE_GYRO_BIAS),
        ):
            if imu_bp.has_attribute(attribute):
                imu_bp.set_attribute(attribute, str(float(value)))

        imu_transform = carla.Transform(
            carla.Location(
                x=IMU_X,
                y=IMU_Y,
                z=IMU_Z,
            ),
            carla.Rotation(
                pitch=IMU_PITCH_DEG,
                yaw=IMU_YAW_DEG,
                roll=IMU_ROLL_DEG,
            ),
        )

        return world.spawn_actor(
            imu_bp,
            imu_transform,
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )

    @staticmethod
    def _on_imu(weak_self, data):
        self = weak_self()
        if self is None:
            return

        self.frame = int(data.frame)
        self.timestamp = float(data.timestamp)

        self.yaw_rate_rad_s = float(data.gyroscope.z)

        self.raw_imu_accel_x_mps2 = float(data.accelerometer.x)
        self.raw_imu_accel_y_mps2 = float(data.accelerometer.y)
        self.raw_imu_accel_z_mps2 = float(data.accelerometer.z)

        self.ready = True

    def get_speed_real_equiv_mps(self):
        velocity = self.parent.get_velocity()

        # Ground-vehicle speed: chá»‰ láº¥y world XY.
        # KhÃ´ng dÃ¹ng vz Ä‘á»ƒ trÃ¡nh gravity/fall speed Ä‘i vÃ o PPO observation.
        speed_carla_mps = math.sqrt(
            velocity.x ** 2
            + velocity.y ** 2
        )

        return float(speed_carla_mps / CARLA_SCALE)

    def get_longitudinal_accel_real_equiv_mps2(self):
        acceleration = self.parent.get_acceleration()
        forward = self.parent.get_transform().get_forward_vector()

        accel_long_carla = (
            acceleration.x * forward.x
            + acceleration.y * forward.y
            + acceleration.z * forward.z
        )

        return float(accel_long_carla / CARLA_SCALE)

    def get_state(self):
        return {
            "speed_mps": self.get_speed_real_equiv_mps(),
            "yaw_rate_rad_s": float(self.yaw_rate_rad_s),
            "longitudinal_accel_mps2":
                self.get_longitudinal_accel_real_equiv_mps2(),
            "imu_ready": bool(self.ready),
            "imu_frame": int(self.frame),
            "imu_timestamp": float(self.timestamp),
        }



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
