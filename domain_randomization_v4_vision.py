# -*- coding: utf-8 -*-
"""
Domain Randomization V4 - CAMERA / VISION / SENSOR extension

Adds a separate experimental DR layer for a SECOND PPO policy:
- camera x/y/z + pitch/yaw/roll per episode
- camera FOV per episode
- weather / lighting per episode
- camera-output augmentation BEFORE the frozen VAE
- experimental speed/yaw/ax observation noise

The proprioceptive noise ranges are conservative placeholders until REAL
sensor-noise measurements are finalized.

Python 3.7 compatible.
"""

from __future__ import print_function

import random
import numpy as np
from PIL import Image, ImageFilter

from vehicle_specs_v3 import CARLA_GEOMETRY_SCALE


CAMERA_REAL_OFFSET_RANGES_M = {
    "x": (-0.010, +0.010),
    "y": (-0.005, +0.005),
    "z": (-0.010, +0.010),
}

CAMERA_ANGLE_OFFSET_RANGES_DEG = {
    "pitch": (-1.5, +1.5),
    "yaw": (-1.0, +1.0),
    "roll": (-0.5, +0.5),
}

CAMERA_FOV_OFFSET_DEG = (-2.0, +2.0)

WEATHER_RANGES = {
    "cloudiness": (0.0, 70.0),
    "precipitation": (0.0, 10.0),
    "wetness": (0.0, 35.0),
    "fog_density": (0.0, 4.0),
    "wind_intensity": (0.0, 20.0),
    "sun_altitude_angle": (35.0, 90.0),
    "sun_azimuth_angle": (0.0, 360.0),
}

IMAGE_EPISODE_RANGES = {
    "brightness_gain": (0.85, 1.15),
    "contrast_gain": (0.85, 1.15),
    "gamma": (0.90, 1.10),
    "red_gain": (0.96, 1.04),
    "green_gain": (0.96, 1.04),
    "blue_gain": (0.96, 1.04),
    "gaussian_noise_sigma": (0.0, 4.0),
}

BLUR_FRAME_PROBABILITY = 0.08
BLUR_RADIUS_RANGE = (0.35, 0.80)

OCCLUSION_FRAME_PROBABILITY = 0.02
OCCLUSION_MAX_AREA_FRACTION = 0.03

STATE_SENSOR_RANGES = {
    "speed_bias_mps": (-0.004, +0.004),
    "speed_noise_std_mps": (0.001, 0.006),
    "yaw_bias_rad_s": (-0.008, +0.008),
    "yaw_noise_std_rad_s": (0.002, 0.012),
    "ax_bias_mps2": (-0.020, +0.020),
    "ax_noise_std_mps2": (0.005, 0.040),
}


def _uniform(rng, pair):
    return float(rng.uniform(float(pair[0]), float(pair[1])))


def sample_vision_domain_v4(seed=None, rng=None):
    if rng is None:
        rng = random.Random(seed)

    p = {
        "camera_dx_real_m": _uniform(rng, CAMERA_REAL_OFFSET_RANGES_M["x"]),
        "camera_dy_real_m": _uniform(rng, CAMERA_REAL_OFFSET_RANGES_M["y"]),
        "camera_dz_real_m": _uniform(rng, CAMERA_REAL_OFFSET_RANGES_M["z"]),
        "camera_pitch_offset_deg": _uniform(
            rng, CAMERA_ANGLE_OFFSET_RANGES_DEG["pitch"]
        ),
        "camera_yaw_offset_deg": _uniform(
            rng, CAMERA_ANGLE_OFFSET_RANGES_DEG["yaw"]
        ),
        "camera_roll_offset_deg": _uniform(
            rng, CAMERA_ANGLE_OFFSET_RANGES_DEG["roll"]
        ),
        "camera_fov_offset_deg": _uniform(rng, CAMERA_FOV_OFFSET_DEG),
    }

    p["camera_dx_carla_m"] = p["camera_dx_real_m"] * CARLA_GEOMETRY_SCALE
    p["camera_dy_carla_m"] = p["camera_dy_real_m"] * CARLA_GEOMETRY_SCALE
    p["camera_dz_carla_m"] = p["camera_dz_real_m"] * CARLA_GEOMETRY_SCALE

    for key, pair in WEATHER_RANGES.items():
        p[key] = _uniform(rng, pair)

    for key, pair in IMAGE_EPISODE_RANGES.items():
        p[key] = _uniform(rng, pair)

    for key, pair in STATE_SENSOR_RANGES.items():
        p[key] = _uniform(rng, pair)

    p["episode_noise_seed"] = int(rng.randrange(0, 2 ** 31 - 1))
    p["vision_dr_status"] = "V4_EXPERIMENTAL"
    p["sensor_noise_status"] = (
        "CONSERVATIVE_PLACEHOLDER_PENDING_REAL_MEASUREMENT"
    )
    return p


def nominal_vision_domain_v4():
    return {
        "camera_dx_real_m": 0.0,
        "camera_dy_real_m": 0.0,
        "camera_dz_real_m": 0.0,
        "camera_dx_carla_m": 0.0,
        "camera_dy_carla_m": 0.0,
        "camera_dz_carla_m": 0.0,
        "camera_pitch_offset_deg": 0.0,
        "camera_yaw_offset_deg": 0.0,
        "camera_roll_offset_deg": 0.0,
        "camera_fov_offset_deg": 0.0,
        "brightness_gain": 1.0,
        "contrast_gain": 1.0,
        "gamma": 1.0,
        "red_gain": 1.0,
        "green_gain": 1.0,
        "blue_gain": 1.0,
        "gaussian_noise_sigma": 0.0,
        "speed_bias_mps": 0.0,
        "speed_noise_std_mps": 0.0,
        "yaw_bias_rad_s": 0.0,
        "yaw_noise_std_rad_s": 0.0,
        "ax_bias_mps2": 0.0,
        "ax_noise_std_mps2": 0.0,
        "episode_noise_seed": 0,
        "vision_dr_status": "OFF",
        "sensor_noise_status": "OFF",
    }


def augment_camera_rgb_v4(image_rgb, params, np_rng):
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "V4 camera image must be HxWx3 RGB, got {}".format(image.shape)
        )

    x = image.astype(np.float32) / 255.0

    x *= float(params["brightness_gain"])

    mean = x.mean(axis=(0, 1), keepdims=True)
    x = mean + float(params["contrast_gain"]) * (x - mean)

    x = np.clip(x, 0.0, 1.0)
    gamma = max(0.25, float(params["gamma"]))
    x = np.power(x, gamma)

    gains = np.asarray(
        [
            float(params["red_gain"]),
            float(params["green_gain"]),
            float(params["blue_gain"]),
        ],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    x *= gains

    sigma_255 = float(params["gaussian_noise_sigma"])
    if sigma_255 > 0.0:
        x += np_rng.normal(
            0.0,
            sigma_255 / 255.0,
            size=x.shape,
        ).astype(np.float32)

    x = np.clip(x, 0.0, 1.0)
    out = (x * 255.0 + 0.5).astype(np.uint8)

    if float(np_rng.rand()) < BLUR_FRAME_PROBABILITY:
        radius = float(
            np_rng.uniform(BLUR_RADIUS_RANGE[0], BLUR_RADIUS_RANGE[1])
        )
        out = np.asarray(
            Image.fromarray(out).filter(
                ImageFilter.GaussianBlur(radius=radius)
            )
        ).copy()

    if float(np_rng.rand()) < OCCLUSION_FRAME_PROBABILITY:
        # V4_READONLY_IMAGE_FIX
        if not out.flags.writeable:
            out = out.copy()
        h, w = out.shape[:2]
        max_area = max(
            1,
            int(h * w * float(OCCLUSION_MAX_AREA_FRACTION)),
        )

        patch_w = max(1, int(np_rng.uniform(0.04 * w, 0.14 * w)))
        patch_h = max(1, int(max_area / float(patch_w)))
        patch_h = min(patch_h, max(1, int(0.15 * h)))

        x0 = int(np_rng.randint(0, max(1, w - patch_w + 1)))
        y0 = int(np_rng.randint(0, max(1, h - patch_h + 1)))
        darken = float(np_rng.uniform(0.45, 0.80))

        patch = out[
            y0:y0 + patch_h,
            x0:x0 + patch_w,
        ].astype(np.float32)

        out[
            y0:y0 + patch_h,
            x0:x0 + patch_w,
        ] = np.clip(
            patch * darken,
            0.0,
            255.0,
        ).astype(np.uint8)

    return np.ascontiguousarray(out)


def noisy_state_for_policy_v4(clean_imu_state, params, np_rng):
    speed = (
        float(clean_imu_state["speed_mps"])
        + float(params["speed_bias_mps"])
        + float(
            np_rng.normal(
                0.0,
                float(params["speed_noise_std_mps"]),
            )
        )
    )

    yaw = (
        float(clean_imu_state["yaw_rate_rad_s"])
        + float(params["yaw_bias_rad_s"])
        + float(
            np_rng.normal(
                0.0,
                float(params["yaw_noise_std_rad_s"]),
            )
        )
    )

    ax = (
        float(clean_imu_state["longitudinal_accel_mps2"])
        + float(params["ax_bias_mps2"])
        + float(
            np_rng.normal(
                0.0,
                float(params["ax_noise_std_mps2"]),
            )
        )
    )

    return {
        "speed_mps": max(0.0, float(speed)),
        "yaw_rate_rad_s": float(yaw),
        "longitudinal_accel_mps2": float(ax),
    }


def print_v4_ranges():
    print("=" * 96)
    print("DOMAIN RANDOMIZATION V4 - VISION / CAMERA / SENSOR")
    print("=" * 96)
    print("Camera REAL XYZ offset:", CAMERA_REAL_OFFSET_RANGES_M)
    print("Camera angle offset:", CAMERA_ANGLE_OFFSET_RANGES_DEG)
    print("Camera FOV offset:", CAMERA_FOV_OFFSET_DEG)
    print("Weather:", WEATHER_RANGES)
    print("Image:", IMAGE_EPISODE_RANGES)
    print("Sensor state:", STATE_SENSOR_RANGES)
    print("Blur probability/frame:", BLUR_FRAME_PROBABILITY)
    print("Occlusion probability/frame:", OCCLUSION_FRAME_PROBABILITY)
    print("=" * 96)


if __name__ == "__main__":
    print_v4_ranges()
    rng = random.Random(6404)
    for idx in range(3):
        print(
            "sample {}: {}".format(
                idx + 1,
                sample_vision_domain_v4(rng=rng),
            )
        )
