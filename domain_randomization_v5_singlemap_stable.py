# -*- coding: utf-8 -*-
"""
Domain Randomization V5 SINGLE-MAP STABLE - CONTROLLED / MILD

Gộp logic production đang dùng từ:
- domain_randomization_v3.py
- domain_randomization_runtime_v3.py
- domain_randomization_v4_vision.py

Mục tiêu:
- Một module DR V5 duy nhất.
- Không import V3/V4.
- Giữ nguyên các range/behavior hiện tại.
- Dynamics DR + steering gain + command delay.
- Camera / weather / image augmentation.
- Policy sensor noise.
- Python 3.7 compatible.

IMPORTANT:
- Longitudinal dynamics vẫn PROVISIONAL vì REAL ax chưa chốt.
- Sample episode-level parameters ONCE per episode.
- Image augmentation chỉ nên chạy ONCE cho mỗi camera frame mới,
  không random lại ở từng control tick 50 Hz.
"""

from __future__ import print_function

import random

import numpy as np
from PIL import Image, ImageFilter

from vehicle_specs_v5 import CARLA_GEOMETRY_SCALE


# =============================================================================
# Helpers
# =============================================================================

def _clip(value, low, high):
    return float(max(float(low), min(float(high), float(value))))


def _uniform(rng, pair):
    return float(rng.uniform(float(pair[0]), float(pair[1])))


# =============================================================================
# Dynamics / timing DR
# =============================================================================

CONTROL_HZ = 50.0
CONTROL_DT_S = 0.020
CAMERA_FPS = 30.0

TORQUE_SCALE_NOMINAL = 0.80
MAX_BRAKE_TORQUE_NOMINAL = 560.0

# SIMPLE baseline intentionally removes the old hard-coded ~30% left/right
# asymmetry. If later real steering calibration proves an asymmetry, put the
# measured values back here and retrain from a fresh PPO.
STEER_GAIN_POSITIVE_NOMINAL = 1.0
STEER_GAIN_NEGATIVE_NOMINAL = 1.0

AUTOBOX_NOMINAL = True
GEAR_SWITCH_TIME_NOMINAL_S = 0.0


DOMAIN_RANDOMIZATION_RANGES = {
    # Longitudinal - PROVISIONAL
    "torque_scale": (
        TORQUE_SCALE_NOMINAL * 0.97,
        TORQUE_SCALE_NOMINAL * 1.03,
    ),
    "max_brake_torque": (
        MAX_BRAKE_TORQUE_NOMINAL * 0.95,
        MAX_BRAKE_TORQUE_NOMINAL * 1.05,
    ),

    # Lateral - calibrated
    "steer_gain_positive": (
        STEER_GAIN_POSITIVE_NOMINAL * 0.97,
        STEER_GAIN_POSITIVE_NOMINAL * 1.03,
    ),
    "steer_gain_negative": (
        STEER_GAIN_NEGATIVE_NOMINAL * 0.97,
        STEER_GAIN_NEGATIVE_NOMINAL * 1.03,
    ),

    # Timing
    "extra_command_delay_ticks": (0, 1),
}

# Camera already has natural 30-FPS vs 50-Hz lag. Keep only a small amount of
# extra actuator-delay DR so total effective delay is not dominated by injection.
EXTRA_DELAY_ONE_TICK_PROBABILITY = 0.00


PENDING_RANDOMIZATION = {
    "longitudinal_accel_noise": "PENDING_REAL_AX",
    "gyro_noise_std": "PENDING_SENSOR_NOISE_ISOLATION",
    "mass": "KEEP_NOMINAL",
    "drag_coefficient": "KEEP_NOMINAL",
    "tire_friction": "KEEP_NOMINAL",
    "center_of_mass": "KEEP_NOMINAL",
    "wheel_radius": "KEEP_MEASURED_GEOMETRY",
}


def sample_domain_randomization_v5(seed=None, rng=None):
    """
    Sample dynamics/timing DR ONCE per episode.
    """
    if rng is None:
        rng = random.Random(seed)

    delay_ticks = 1 if (
        rng.random() < EXTRA_DELAY_ONE_TICK_PROBABILITY
    ) else 0

    return {
        "torque_scale": _uniform(
            rng,
            DOMAIN_RANDOMIZATION_RANGES["torque_scale"],
        ),
        "max_brake_torque": _uniform(
            rng,
            DOMAIN_RANDOMIZATION_RANGES["max_brake_torque"],
        ),
        "steer_gain_positive": _uniform(
            rng,
            DOMAIN_RANDOMIZATION_RANGES["steer_gain_positive"],
        ),
        "steer_gain_negative": _uniform(
            rng,
            DOMAIN_RANDOMIZATION_RANGES["steer_gain_negative"],
        ),
        "extra_command_delay_ticks": int(delay_ticks),

        "control_hz": CONTROL_HZ,
        "control_dt_s": CONTROL_DT_S,
        "camera_fps": CAMERA_FPS,
        "use_gear_autobox": AUTOBOX_NOMINAL,
        "gear_switch_time_s": GEAR_SWITCH_TIME_NOMINAL_S,

        "longitudinal_status": "PROVISIONAL_PENDING_REAL_AX",
        "lateral_status": "SYMMETRIC_MILD_DR",
        "timing_status": "NO_EXTRA_DELAY_DR",
    }


def nominal_domain_parameters_v5():
    """
    Exact non-randomized dynamics/timing parameters.
    """
    return {
        "torque_scale": TORQUE_SCALE_NOMINAL,
        "max_brake_torque": MAX_BRAKE_TORQUE_NOMINAL,
        "steer_gain_positive": STEER_GAIN_POSITIVE_NOMINAL,
        "steer_gain_negative": STEER_GAIN_NEGATIVE_NOMINAL,
        "extra_command_delay_ticks": 0,
        "control_hz": CONTROL_HZ,
        "control_dt_s": CONTROL_DT_S,
        "camera_fps": CAMERA_FPS,
        "use_gear_autobox": AUTOBOX_NOMINAL,
        "gear_switch_time_s": GEAR_SWITCH_TIME_NOMINAL_S,
        "longitudinal_status": "PROVISIONAL_PENDING_REAL_AX",
        "lateral_status": "SYMMETRIC_MILD_DR",
        "timing_status": "NO_EXTRA_DELAY_DR",
    }


def apply_domain_physics_v5(vehicle, params, carla):
    """
    Apply episode-level dynamics DR to one freshly spawned CARLA vehicle.

    Returns readback values for logging/verification.
    """
    physics = vehicle.get_physics_control()

    torque_scale = float(params["torque_scale"])
    physics.torque_curve = [
        carla.Vector2D(
            float(point.x),
            float(point.y) * torque_scale,
        )
        for point in list(physics.torque_curve)
    ]

    wheels = list(physics.wheels)
    brake_torque = float(params["max_brake_torque"])

    for wheel in wheels:
        wheel.max_brake_torque = brake_torque

    physics.wheels = wheels
    physics.use_gear_autobox = bool(params["use_gear_autobox"])
    physics.gear_switch_time = float(params["gear_switch_time_s"])

    vehicle.apply_physics_control(physics)

    applied = vehicle.get_physics_control()

    return {
        "torque_curve": [
            (float(p.x), float(p.y))
            for p in list(applied.torque_curve)
        ],
        "max_brake_torque": [
            float(w.max_brake_torque)
            for w in list(applied.wheels)
        ],
        "use_gear_autobox": bool(applied.use_gear_autobox),
        "gear_switch_time_s": float(applied.gear_switch_time),
    }


class DomainRandomizedActionControllerV5(object):
    """
    Wrap ActionControllerV5 with:
    - asymmetric calibrated steer gain
    - optional 0/1 control-tick command delay

    Observation/reward command semantics remain in REAL logical command space.
    """

    def __init__(
        self,
        base_controller,
        steer_gain_positive,
        steer_gain_negative,
        extra_command_delay_ticks=0,
    ):
        self.base_controller = base_controller
        self.steer_gain_positive = float(steer_gain_positive)
        self.steer_gain_negative = float(steer_gain_negative)

        self.extra_command_delay_ticks = int(extra_command_delay_ticks)
        if self.extra_command_delay_ticks not in (0, 1):
            raise ValueError(
                "V5 chỉ hỗ trợ extra_command_delay_ticks = 0 hoặc 1."
            )

        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0
        self._delay_queue = []

    def _reset_delay_queue(self):
        self._delay_queue = [
            (0.0, 0.0)
            for _ in range(self.extra_command_delay_ticks)
        ]

    def reset(self):
        self.base_controller.reset()
        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0
        self._reset_delay_queue()

    def _physical_steer(self, logical_steer):
        logical_steer = _clip(logical_steer, -1.0, 1.0)

        if logical_steer > 0.0:
            physical = logical_steer * self.steer_gain_positive
        elif logical_steer < 0.0:
            physical = logical_steer * self.steer_gain_negative
        else:
            physical = 0.0

        return _clip(physical, -1.0, 1.0)

    def _apply_delay(self, steer_cmd, speed_cmd_mps):
        incoming = (
            _clip(steer_cmd, -1.0, 1.0),
            max(0.0, float(speed_cmd_mps)),
        )

        if self.extra_command_delay_ticks == 0:
            return incoming

        self._delay_queue.append(incoming)
        return self._delay_queue.pop(0)

    def step(self, steer_cmd, speed_cmd_mps, dt):
        policy_requested_steer = _clip(steer_cmd, -1.0, 1.0)
        policy_requested_speed = max(0.0, float(speed_cmd_mps))

        logical_steer, logical_speed = self._apply_delay(
            policy_requested_steer,
            policy_requested_speed,
        )

        physical_steer = self._physical_steer(logical_steer)

        base_info = self.base_controller.step(
            steer_cmd=physical_steer,
            speed_cmd_mps=logical_speed,
            dt=dt,
        )

        self.prev_steer_cmd = float(logical_steer)
        self.prev_speed_cmd_mps = float(logical_speed)

        info = dict(base_info)

        info["requested_steer_cmd"] = float(policy_requested_steer)
        info["requested_speed_cmd_mps"] = float(policy_requested_speed)

        info["actual_steer_cmd"] = float(logical_steer)
        info["actual_speed_cmd_mps"] = float(logical_speed)

        info["dr_physical_steer_cmd"] = float(physical_steer)
        info["dr_carla_actual_steer_cmd"] = float(
            base_info.get("actual_steer_cmd", physical_steer)
        )
        info["dr_extra_command_delay_ticks"] = int(
            self.extra_command_delay_ticks
        )
        info["dr_steer_gain_positive"] = float(
            self.steer_gain_positive
        )
        info["dr_steer_gain_negative"] = float(
            self.steer_gain_negative
        )

        return info


# =============================================================================
# Camera / vision / sensor DR
# =============================================================================

CAMERA_REAL_OFFSET_RANGES_M = {
    "x": (-0.004, +0.004),
    "y": (-0.003, +0.003),
    "z": (-0.004, +0.004),
}

CAMERA_ANGLE_OFFSET_RANGES_DEG = {
    "pitch": (-0.6, +0.6),
    "yaw": (-0.5, +0.5),
    "roll": (-0.2, +0.2),
}

CAMERA_FOV_OFFSET_DEG = (-1.0, +1.0)

WEATHER_RANGES = {
    "cloudiness": (0.0, 35.0),
    "precipitation": (0.0, 3.0),
    "wetness": (0.0, 10.0),
    "fog_density": (0.0, 1.0),
    "wind_intensity": (0.0, 10.0),
    "sun_altitude_angle": (45.0, 85.0),
    "sun_azimuth_angle": (0.0, 360.0),
}

IMAGE_EPISODE_RANGES = {
    "brightness_gain": (0.94, 1.06),
    "contrast_gain": (0.94, 1.06),
    "gamma": (0.97, 1.03),
    "red_gain": (0.985, 1.015),
    "green_gain": (0.985, 1.015),
    "blue_gain": (0.985, 1.015),
    "gaussian_noise_sigma": (0.0, 1.5),
}

BLUR_FRAME_PROBABILITY = 0.02
BLUR_RADIUS_RANGE = (0.25, 0.50)

OCCLUSION_FRAME_PROBABILITY = 0.00
OCCLUSION_MAX_AREA_FRACTION = 0.03

STATE_SENSOR_RANGES = {
    "speed_bias_mps": (-0.002, +0.002),
    "speed_noise_std_mps": (0.0005, 0.0020),
    "yaw_bias_rad_s": (-0.003, +0.003),
    "yaw_noise_std_rad_s": (0.001, 0.004),
    "ax_bias_mps2": (-0.010, +0.010),
    "ax_noise_std_mps2": (0.003, 0.015),
}


def sample_vision_domain_v5(seed=None, rng=None):
    """
    Sample camera/weather/image/sensor DR ONCE per episode.
    """
    if rng is None:
        rng = random.Random(seed)

    p = {
        "camera_dx_real_m": _uniform(
            rng, CAMERA_REAL_OFFSET_RANGES_M["x"]
        ),
        "camera_dy_real_m": _uniform(
            rng, CAMERA_REAL_OFFSET_RANGES_M["y"]
        ),
        "camera_dz_real_m": _uniform(
            rng, CAMERA_REAL_OFFSET_RANGES_M["z"]
        ),
        "camera_pitch_offset_deg": _uniform(
            rng, CAMERA_ANGLE_OFFSET_RANGES_DEG["pitch"]
        ),
        "camera_yaw_offset_deg": _uniform(
            rng, CAMERA_ANGLE_OFFSET_RANGES_DEG["yaw"]
        ),
        "camera_roll_offset_deg": _uniform(
            rng, CAMERA_ANGLE_OFFSET_RANGES_DEG["roll"]
        ),
        "camera_fov_offset_deg": _uniform(
            rng, CAMERA_FOV_OFFSET_DEG
        ),
    }

    p["camera_dx_carla_m"] = (
        p["camera_dx_real_m"] * float(CARLA_GEOMETRY_SCALE)
    )
    p["camera_dy_carla_m"] = (
        p["camera_dy_real_m"] * float(CARLA_GEOMETRY_SCALE)
    )
    p["camera_dz_carla_m"] = (
        p["camera_dz_real_m"] * float(CARLA_GEOMETRY_SCALE)
    )

    for key, pair in WEATHER_RANGES.items():
        p[key] = _uniform(rng, pair)

    for key, pair in IMAGE_EPISODE_RANGES.items():
        p[key] = _uniform(rng, pair)

    for key, pair in STATE_SENSOR_RANGES.items():
        p[key] = _uniform(rng, pair)

    p["episode_noise_seed"] = int(
        rng.randrange(0, 2 ** 31 - 1)
    )
    p["vision_dr_status"] = "V5"
    p["sensor_noise_status"] = (
        "CONSERVATIVE_PLACEHOLDER_PENDING_REAL_MEASUREMENT"
    )

    return p


def nominal_vision_domain_v5():
    """
    Exact no-randomization camera/vision/sensor configuration.
    """
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


def augment_camera_rgb_v5(image_rgb, params, np_rng):
    """
    Apply RGB augmentation to one camera frame.

    Caller should cache output by camera frame id so the same 30-FPS
    frame reused by 50-Hz control is not augmented twice.
    """
    image = np.asarray(image_rgb)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "V5 camera image must be HxWx3 RGB, got {}".format(
                image.shape
            )
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
            np_rng.uniform(
                BLUR_RADIUS_RANGE[0],
                BLUR_RADIUS_RANGE[1],
            )
        )
        out = np.asarray(
            Image.fromarray(out).filter(
                ImageFilter.GaussianBlur(radius=radius)
            )
        ).copy()

    if float(np_rng.rand()) < OCCLUSION_FRAME_PROBABILITY:
        if not out.flags.writeable:
            out = out.copy()

        h, w = out.shape[:2]

        max_area = max(
            1,
            int(
                h
                * w
                * float(OCCLUSION_MAX_AREA_FRACTION)
            ),
        )

        patch_w = max(
            1,
            int(np_rng.uniform(0.04 * w, 0.14 * w)),
        )
        patch_h = max(
            1,
            int(max_area / float(patch_w)),
        )
        patch_h = min(
            patch_h,
            max(1, int(0.15 * h)),
        )

        x0 = int(
            np_rng.randint(
                0,
                max(1, w - patch_w + 1),
            )
        )
        y0 = int(
            np_rng.randint(
                0,
                max(1, h - patch_h + 1),
            )
        )

        darken = float(
            np_rng.uniform(0.45, 0.80)
        )

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


def noisy_state_for_policy_v5(clean_imu_state, params, np_rng):
    """
    Add observation-only sensor bias/noise.

    Reward should still use clean CARLA state.
    """
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


# =============================================================================
# Diagnostics
# =============================================================================

def print_domain_randomization_v5_config():
    print("=" * 96)
    print("DOMAIN RANDOMIZATION V5 SIMPLE - SAFE BASELINE")
    print("=" * 96)

    print("DYNAMICS NOMINAL")
    for key, value in nominal_domain_parameters_v5().items():
        print("  {:32s}: {}".format(key, value))

    print("")
    print("DYNAMICS RANGES")
    for key, value in DOMAIN_RANDOMIZATION_RANGES.items():
        print("  {:32s}: {}".format(key, value))

    print("")
    print(
        "  {:32s}: {:.0%}".format(
            "P(extra_delay = 1 tick)",
            EXTRA_DELAY_ONE_TICK_PROBABILITY,
        )
    )

    print("")
    print("CAMERA REAL XYZ OFFSET")
    print(" ", CAMERA_REAL_OFFSET_RANGES_M)

    print("CAMERA ANGLE OFFSET")
    print(" ", CAMERA_ANGLE_OFFSET_RANGES_DEG)

    print("CAMERA FOV OFFSET")
    print(" ", CAMERA_FOV_OFFSET_DEG)

    print("WEATHER")
    print(" ", WEATHER_RANGES)

    print("IMAGE")
    print(" ", IMAGE_EPISODE_RANGES)

    print("STATE SENSOR")
    print(" ", STATE_SENSOR_RANGES)

    print("BLUR PROBABILITY/FRAME:", BLUR_FRAME_PROBABILITY)
    print("OCCLUSION PROBABILITY/FRAME:", OCCLUSION_FRAME_PROBABILITY)

    print("")
    print("PENDING / KEEP FIXED")
    for key, value in PENDING_RANDOMIZATION.items():
        print("  {:32s}: {}".format(key, value))

    print("=" * 96)


if __name__ == "__main__":
    print_domain_randomization_v5_config()

    print("")
    print("3 deterministic dynamics samples")
    rng = random.Random(6106)
    for idx in range(3):
        print(
            "  sample {:02d}: {}".format(
                idx + 1,
                sample_domain_randomization_v5(rng=rng),
            )
        )

    print("")
    print("3 deterministic vision samples")
    rng = random.Random(6404)
    for idx in range(3):
        print(
            "  sample {:02d}: {}".format(
                idx + 1,
                sample_vision_domain_v5(rng=rng),
            )
        )
