# -*- coding: utf-8 -*-
"""
GĐ6 - Domain Randomization V3
=============================

Mục tiêu:
- Định nghĩa RANGE randomization dựa trên các sai số REAL <-> SIM đã đo.
- CHƯA tích hợp vào PPO environment ở file này.
- CHƯA freeze longitudinal vì REAL ax còn pending.
- Không random "mạnh tay"; ưu tiên khoảng hẹp, có cơ sở đo.

Nominal hiện tại:
- control = 50 Hz, dt = 0.020 s
- camera = 30 FPS
- extra command latency nominal = 0 ms
- longitudinal provisional:
    torque_scale = 0.80
    max_brake_torque = 560
    autobox = True
    gear_switch_time = 0.0 s
- lateral calibrated:
    steer_gain_positive = 1.1245
    steer_gain_negative = 1.4639

Measured residuals used to define ranges:
- steady speed mostly within about +/-5%
- full-stop timing about +/-11%
- lateral yaw after calibration within about +/-11.3%
- communication timing unresolved below one 20 ms control tick
- REAL ax is not finalized -> longitudinal DR marked PROVISIONAL

Python 3.7 compatible.
"""

from __future__ import print_function

import random


# =============================================================================
# Nominal values
# =============================================================================

CONTROL_HZ = 50.0
CONTROL_DT_S = 0.020
CAMERA_FPS = 30.0

TORQUE_SCALE_NOMINAL = 0.80
MAX_BRAKE_TORQUE_NOMINAL = 560.0

STEER_GAIN_POSITIVE_NOMINAL = 1.1245
STEER_GAIN_NEGATIVE_NOMINAL = 1.4639

AUTOBOX_NOMINAL = True
GEAR_SWITCH_TIME_NOMINAL_S = 0.0


# =============================================================================
# GĐ6 randomization ranges
# =============================================================================
#
# IMPORTANT:
# Longitudinal ranges are deliberately conservative because REAL ax is pending.
# After ax is fixed, only this section needs to be updated before GĐ7 freeze.
#

DOMAIN_RANDOMIZATION_RANGES = {
    # -------------------------------------------------------------------------
    # Longitudinal - PROVISIONAL
    # -------------------------------------------------------------------------
    # +/-7.5% around torque nominal.
    # Enough to cover the remaining steady/accel mismatch without letting
    # training see unrealistically weak/strong drivetrains.
    "torque_scale": (
        TORQUE_SCALE_NOMINAL * 0.925,
        TORQUE_SCALE_NOMINAL * 1.075,
    ),

    # Full-stop timing residual was roughly +/-11%.
    # Use +/-12% around nominal brake torque.
    "max_brake_torque": (
        MAX_BRAKE_TORQUE_NOMINAL * 0.88,
        MAX_BRAKE_TORQUE_NOMINAL * 1.12,
    ),

    # -------------------------------------------------------------------------
    # Lateral - measured/calibrated
    # -------------------------------------------------------------------------
    # Post-calibration yaw residual stayed inside about +/-11.3%.
    # Use +/-12% multiplicative range independently by steering direction.
    "steer_gain_positive": (
        STEER_GAIN_POSITIVE_NOMINAL * 0.88,
        STEER_GAIN_POSITIVE_NOMINAL * 1.12,
    ),
    "steer_gain_negative": (
        STEER_GAIN_NEGATIVE_NOMINAL * 0.88,
        STEER_GAIN_NEGATIVE_NOMINAL * 1.12,
    ),

    # -------------------------------------------------------------------------
    # Timing - measured at 50 Hz resolution
    # -------------------------------------------------------------------------
    # Nominal remains 0 additional ticks.
    # REAL logging only resolves latency to <= 1 control tick,
    # so DR may occasionally inject one extra tick to cover that uncertainty.
    "extra_command_delay_ticks": (0, 1),

    # No artificial camera delay here:
    # 30 FPS camera in 50 Hz control already naturally causes frame reuse/age.
}


# Probability of using one extra command-delay tick.
# Keep nominal behavior dominant.
EXTRA_DELAY_ONE_TICK_PROBABILITY = 0.25


# =============================================================================
# Disabled / pending randomization
# =============================================================================

PENDING_RANDOMIZATION = {
    # REAL ax not finalized.
    "longitudinal_accel_noise": "PENDING_REAL_AX",

    # Current yaw std contains vehicle dynamics + road response, so it should
    # not be blindly interpreted as pure gyro sensor noise.
    "gyro_noise_std": "PENDING_SENSOR_NOISE_ISOLATION",

    # No isolated REAL measurement yet for these.
    "mass": "KEEP_NOMINAL",
    "drag_coefficient": "KEEP_NOMINAL",
    "tire_friction": "KEEP_NOMINAL",
    "center_of_mass": "KEEP_NOMINAL",
    "wheel_radius": "KEEP_MEASURED_GEOMETRY",
}


# =============================================================================
# Sampler
# =============================================================================

def _uniform(rng, pair):
    return float(rng.uniform(float(pair[0]), float(pair[1])))


def sample_domain_randomization(seed=None, rng=None):
    """
    Return one episode-level domain-randomization sample.

    Recommended usage later:
        params = sample_domain_randomization(rng=episode_rng)

    Sample ONCE per episode, not every 20 ms tick.
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

        # Frozen nominal timing / drivetrain choices.
        "control_hz": CONTROL_HZ,
        "control_dt_s": CONTROL_DT_S,
        "camera_fps": CAMERA_FPS,
        "use_gear_autobox": AUTOBOX_NOMINAL,
        "gear_switch_time_s": GEAR_SWITCH_TIME_NOMINAL_S,

        # Metadata for later logs/checkpoints.
        "longitudinal_status": "PROVISIONAL_PENDING_REAL_AX",
        "lateral_status": "CALIBRATED",
        "timing_status": "FROZEN_50HZ_30FPS",
    }


def nominal_domain_parameters():
    """
    Exact nominal (non-randomized) configuration.
    Useful for evaluation and debugging.
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
        "lateral_status": "CALIBRATED",
        "timing_status": "FROZEN_50HZ_30FPS",
    }


def print_gd6_config():
    print("=" * 88)
    print("GĐ6 - DOMAIN RANDOMIZATION V3")
    print("=" * 88)

    print("NOMINAL")
    for key, value in nominal_domain_parameters().items():
        print("  {:32s}: {}".format(key, value))

    print("")
    print("RANGES")
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
    print("PENDING / KEEP FIXED")
    for key, value in PENDING_RANDOMIZATION.items():
        print("  {:32s}: {}".format(key, value))

    print("=" * 88)


if __name__ == "__main__":
    print_gd6_config()

    print("")
    print("5 deterministic samples (seed=6106)")
    rng = random.Random(6106)

    for idx in range(5):
        print(
            "  sample {:02d}: {}".format(
                idx + 1,
                sample_domain_randomization(rng=rng),
            )
        )
