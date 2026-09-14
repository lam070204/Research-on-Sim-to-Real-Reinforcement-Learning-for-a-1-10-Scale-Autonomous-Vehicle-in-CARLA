# -*- coding: utf-8 -*-
"""
CARLA RGB V4 environment extension.

Subclasses the CURRENT V3 environment and adds:
- camera mount/FOV randomization per episode
- weather/lighting randomization per episode
- camera-output augmentation BEFORE frozen VAE
- experimental noisy speed/yaw/ax ONLY for policy observation

Keeps:
- current V3 dynamics DR
- reward implementation
- obs = 100
- action = 2
- clean CARLA state for reward
"""

from __future__ import print_function

import copy
import random

import numpy as np

from simulation.carla_connection_v3 import carla
from simulation.carla_environment_rgb_v3 import (
    CarlaEnvironmentRGB as CarlaEnvironmentRGBV3Base,
)
from simulation.carla_sensors_v3 import (
    CameraSensor,
    _configure_front_rgb_blueprint,
)
from vehicle_specs_v3 import (
    FRONT_CAMERA_X,
    FRONT_CAMERA_Y,
    FRONT_CAMERA_Z,
    FRONT_CAMERA_PITCH_DEG,
    FRONT_CAMERA_YAW_DEG,
    FRONT_CAMERA_ROLL_DEG,
    FRONT_CAMERA_FOV_DEG,
    FRONT_CAMERA_FPS,
)

from domain_randomization_v4_vision import (
    sample_vision_domain_v4,
    nominal_vision_domain_v4,
    augment_camera_rgb_v4,
    noisy_state_for_policy_v4,
)


def _safe_destroy_sensor(wrapper):
    if wrapper is None:
        return

    sensor = getattr(wrapper, "sensor", None)
    if sensor is None:
        return

    try:
        sensor.stop()
    except Exception:
        pass

    try:
        if sensor.is_alive:
            sensor.destroy()
    except Exception:
        try:
            sensor.destroy()
        except Exception:
            pass


class CameraSensorVisionDRV4(CameraSensor):
    def __init__(self, vehicle, camera_transform, fov_deg):
        self._v4_camera_transform = camera_transform
        self._v4_fov_deg = float(fov_deg)
        super(CameraSensorVisionDRV4, self).__init__(vehicle)

    def _set_camera_sensor(self, world):
        camera_bp = world.get_blueprint_library().find(self.sensor_name)
        _configure_front_rgb_blueprint(camera_bp)

        camera_bp.set_attribute(
            "fov",
            str(float(self._v4_fov_deg)),
        )
        camera_bp.set_attribute(
            "sensor_tick",
            str(1.0 / float(FRONT_CAMERA_FPS)),
        )

        return world.spawn_actor(
            camera_bp,
            self._v4_camera_transform,
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )


class CarlaEnvironmentRGBVisionDRV4(CarlaEnvironmentRGBV3Base):
    def __init__(self, *args, **kwargs):
        self.vision_dr_enabled = bool(
            kwargs.get("domain_randomization_enabled", True)
        )

        base_seed = kwargs.get("domain_randomization_seed", None)

        if base_seed is None:
            self._vision_rng = random.Random()
        else:
            self._vision_rng = random.Random(int(base_seed) + 44004)

        self._vision_params = nominal_vision_domain_v4()
        self._vision_np_rng = np.random.RandomState(0)

        self.last_v4_augmented_rgb = None
        self.last_v4_policy_state = None

        # V4_CAMERA_NOISE_PER_NEW_FRAME_FIX
        # Camera thật 30 FPS, PPO/control 50 Hz:
        # cùng một camera frame được reuse thì phải reuse luôn
        # cùng một phiên bản augmented, không random noise lại ở 50 Hz.
        self._v4_last_augmented_camera_frame = None

        super(
            CarlaEnvironmentRGBVisionDRV4,
            self,
        ).__init__(*args, **kwargs)

        try:
            self._v4_initial_weather = copy.copy(
                self.world.get_weather()
            )
        except Exception:
            self._v4_initial_weather = None

        print(
            "ENV V4 VISION DR | enabled={}".format(
                self.vision_dr_enabled
            )
        )

    def _sample_v4_episode(self):
        self._v4_last_augmented_camera_frame = None
        self.last_v4_augmented_rgb = None

        if not self.vision_dr_enabled:
            self._vision_params = nominal_vision_domain_v4()
            self._vision_np_rng = np.random.RandomState(0)
            return

        self._vision_params = sample_vision_domain_v4(
            rng=self._vision_rng
        )

        self._vision_np_rng = np.random.RandomState(
            int(self._vision_params["episode_noise_seed"])
        )

    def _make_v4_camera_transform(self):
        p = self._vision_params

        return carla.Transform(
            carla.Location(
                x=(
                    float(FRONT_CAMERA_X)
                    + float(p["camera_dx_carla_m"])
                ),
                y=(
                    float(FRONT_CAMERA_Y)
                    + float(p["camera_dy_carla_m"])
                ),
                z=(
                    float(FRONT_CAMERA_Z)
                    + float(p["camera_dz_carla_m"])
                ),
            ),
            carla.Rotation(
                pitch=(
                    float(FRONT_CAMERA_PITCH_DEG)
                    + float(p["camera_pitch_offset_deg"])
                ),
                yaw=(
                    float(FRONT_CAMERA_YAW_DEG)
                    + float(p["camera_yaw_offset_deg"])
                ),
                roll=(
                    float(FRONT_CAMERA_ROLL_DEG)
                    + float(p["camera_roll_offset_deg"])
                ),
            ),
        )

    def _apply_v4_weather(self):
        if not self.vision_dr_enabled:
            return

        p = self._vision_params

        try:
            weather = self.world.get_weather()

            for key in (
                "cloudiness",
                "precipitation",
                "wetness",
                "fog_density",
                "wind_intensity",
                "sun_altitude_angle",
                "sun_azimuth_angle",
            ):
                if hasattr(weather, key):
                    setattr(weather, key, float(p[key]))

            self.world.set_weather(weather)

        except Exception as error:
            print(
                "V4 WARNING | weather DR not applied: {}".format(
                    error
                )
            )

    def _spawn_episode_components(self):
        # V4_SINGLE_CAMERA_SPAWN_FIX
        # Giữ nguyên logic V3/GĐ6, nhưng tạm thay CameraSensor
        # để V3 spawn trực tiếp camera V4 randomized.
        if not self.vision_dr_enabled:
            return super(
                CarlaEnvironmentRGBVisionDRV4,
                self,
            )._spawn_episode_components()

        import simulation.carla_environment_rgb_v3 as v3_env_module
        # V4_WHEEL_OFFROAD_MANAGER_FIX
        from reward_manager_v4 import RewardManagerV4

        camera_transform = self._make_v4_camera_transform()

        fov_deg = (
            float(FRONT_CAMERA_FOV_DEG)
            + float(
                self._vision_params[
                    "camera_fov_offset_deg"
                ]
            )
        )

        original_camera_sensor = v3_env_module.CameraSensor
        original_reward_manager = v3_env_module.RewardManagerV3

        def _v4_camera_factory(vehicle):
            return CameraSensorVisionDRV4(
                vehicle=vehicle,
                camera_transform=camera_transform,
                fov_deg=fov_deg,
            )

        v3_env_module.CameraSensor = _v4_camera_factory
        v3_env_module.RewardManagerV3 = RewardManagerV4

        try:
            return super(
                CarlaEnvironmentRGBVisionDRV4,
                self,
            )._spawn_episode_components()
        finally:
            v3_env_module.CameraSensor = original_camera_sensor
            v3_env_module.RewardManagerV3 = original_reward_manager

    def _build_observation(self):
        if not self.vision_dr_enabled:
            return super(
                CarlaEnvironmentRGBVisionDRV4,
                self,
            )._build_observation()

        if not self.camera_obj.front_camera:
            raise RuntimeError("V4 camera has no RGB frame.")

        clean_imu_state = self.imu_obj.get_state()

        if not clean_imu_state["imu_ready"]:
            raise RuntimeError("V4 IMU is not ready.")

        raw_rgb = self.camera_obj.front_camera[-1]

        camera_frame = int(
            getattr(self.camera_obj, "frame", -1)
        )

        # Camera-output augmentation belongs to the CAMERA FRAME,
        # not to the 50-Hz control tick.
        if (
            self.last_v4_augmented_rgb is None
            or self._v4_last_augmented_camera_frame != camera_frame
        ):
            augmented_rgb = augment_camera_rgb_v4(
                image_rgb=raw_rgb,
                params=self._vision_params,
                np_rng=self._vision_np_rng,
            )
            self.last_v4_augmented_rgb = augmented_rgb
            self._v4_last_augmented_camera_frame = camera_frame
        else:
            augmented_rgb = self.last_v4_augmented_rgb

        policy_state = noisy_state_for_policy_v4(
            clean_imu_state=clean_imu_state,
            params=self._vision_params,
            np_rng=self._vision_np_rng,
        )

        observation = self.observation_builder.build(
            image_rgb=augmented_rgb,
            speed_mps=policy_state["speed_mps"],
            yaw_rate_rad_s=policy_state["yaw_rate_rad_s"],
            longitudinal_accel_mps2=policy_state[
                "longitudinal_accel_mps2"
            ],
            prev_steer_cmd=self.action_controller.prev_steer_cmd,
            prev_speed_cmd_mps=(
                self.action_controller.prev_speed_cmd_mps
            ),
        )

        self.last_v4_augmented_rgb = augmented_rgb
        self.last_v4_policy_state = dict(policy_state)

        # Clean state is intentionally returned for reward calculation.
        return observation, clean_imu_state

    def reset(self):
        self._sample_v4_episode()
        self._apply_v4_weather()

        observation = super(
            CarlaEnvironmentRGBVisionDRV4,
            self,
        ).reset()

        if self.vision_dr_enabled:
            p = self._vision_params

            print(
                "V4 VISION DOMAIN | "
                "camREAL xyz=({:+.3f},{:+.3f},{:+.3f}) m | "
                "rpy=({:+.2f},{:+.2f},{:+.2f}) deg | "
                "fov_d={:+.2f} | "
                "bright={:.3f} contrast={:.3f} gamma={:.3f} | "
                "pix_sigma={:.2f}".format(
                    p["camera_dx_real_m"],
                    p["camera_dy_real_m"],
                    p["camera_dz_real_m"],
                    p["camera_roll_offset_deg"],
                    p["camera_pitch_offset_deg"],
                    p["camera_yaw_offset_deg"],
                    p["camera_fov_offset_deg"],
                    p["brightness_gain"],
                    p["contrast_gain"],
                    p["gamma"],
                    p["gaussian_noise_sigma"],
                )
            )

            print(
                "V4 SENSOR OBS | "
                "speed bias/std={:+.4f}/{:.4f} | "
                "yaw bias/std={:+.4f}/{:.4f} | "
                "ax bias/std={:+.4f}/{:.4f}".format(
                    p["speed_bias_mps"],
                    p["speed_noise_std_mps"],
                    p["yaw_bias_rad_s"],
                    p["yaw_noise_std_rad_s"],
                    p["ax_bias_mps2"],
                    p["ax_noise_std_mps2"],
                )
            )

        return observation

    def step(self, action):
        observation, reward, done, info = super(
            CarlaEnvironmentRGBVisionDRV4,
            self,
        ).step(action)

        info["vision_dr_v4"] = dict(self._vision_params)

        if self.last_v4_policy_state is not None:
            info["policy_sensor_state_v4"] = dict(
                self.last_v4_policy_state
            )

        return observation, reward, done, info

    def close(self):
        try:
            super(
                CarlaEnvironmentRGBVisionDRV4,
                self,
            ).close()
        finally:
            if self._v4_initial_weather is not None:
                try:
                    self.world.set_weather(self._v4_initial_weather)
                except Exception:
                    pass


CarlaEnvironmentRGB = CarlaEnvironmentRGBVisionDRV4
CarlaEnvironmentRGBV4 = CarlaEnvironmentRGBVisionDRV4
