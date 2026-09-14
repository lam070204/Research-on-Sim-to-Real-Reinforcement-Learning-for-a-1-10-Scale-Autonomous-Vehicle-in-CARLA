# -*- coding: utf-8 -*-
"""
Offline smoke test for domain_randomization_v4_vision.py.
No CARLA server required.
"""

from __future__ import print_function

import random
import numpy as np

from domain_randomization_v4_vision import (
    sample_vision_domain_v4,
    augment_camera_rgb_v4,
    noisy_state_for_policy_v4,
)


def main():
    rng = random.Random(6404)

    print("=" * 96)
    print("V4 VISION DR OFFLINE SMOKE")
    print("=" * 96)

    params = sample_vision_domain_v4(rng=rng)

    print("sample:")
    for key in sorted(params):
        print("  {:34s}: {}".format(key, params[key]))

    image = np.zeros((80, 160, 3), dtype=np.uint8)
    image[:, :, 0] = 100
    image[:, :, 1] = 130
    image[:, :, 2] = 160

    np_rng = np.random.RandomState(
        int(params["episode_noise_seed"])
    )

    augmented = augment_camera_rgb_v4(
        image_rgb=image,
        params=params,
        np_rng=np_rng,
    )

    if augmented.shape != (80, 160, 3):
        raise RuntimeError(
            "FAIL image shape: {}".format(augmented.shape)
        )

    if augmented.dtype != np.uint8:
        raise RuntimeError(
            "FAIL image dtype: {}".format(augmented.dtype)
        )

    clean = {
        "speed_mps": 0.60,
        "yaw_rate_rad_s": 0.20,
        "longitudinal_accel_mps2": 0.10,
    }

    noisy = noisy_state_for_policy_v4(
        clean_imu_state=clean,
        params=params,
        np_rng=np_rng,
    )

    print("")
    print("clean state:", clean)
    print("noisy policy state:", noisy)
    print(
        "image mean raw/aug: {:.2f} / {:.2f}".format(
            float(image.mean()),
            float(augmented.mean()),
        )
    )

    print("")
    print("RESULT: V4 VISION DR OFFLINE SMOKE PASS")
    print("=" * 96)


if __name__ == "__main__":
    main()
