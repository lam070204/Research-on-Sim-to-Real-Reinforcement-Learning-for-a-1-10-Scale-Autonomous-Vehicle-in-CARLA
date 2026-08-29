# -*- coding: utf-8 -*-
"""
GĐ6 smoke test - domain randomization ranges.

Không cần CARLA.
Chỉ kiểm tra:
- sampler deterministic khi cùng seed
- mọi sample nằm trong range
- delay chỉ 0 hoặc 1 tick
- nominal timing vẫn 50 Hz / 20 ms / camera 30 FPS

Chạy:
    python .\smoke_test_domain_randomization_v3.py
"""

from __future__ import print_function

import random

from domain_randomization_v3 import (
    CONTROL_HZ,
    CONTROL_DT_S,
    CAMERA_FPS,
    DOMAIN_RANDOMIZATION_RANGES,
    nominal_domain_parameters,
    sample_domain_randomization,
)


def in_range(value, pair):
    return float(pair[0]) <= float(value) <= float(pair[1])


def main():
    print("=" * 88)
    print("GĐ6 - DOMAIN RANDOMIZATION SMOKE TEST")
    print("=" * 88)

    assert abs(CONTROL_HZ - 50.0) < 1e-12
    assert abs(CONTROL_DT_S - 0.020) < 1e-12
    assert abs(CAMERA_FPS - 30.0) < 1e-12

    nominal = nominal_domain_parameters()

    assert nominal["extra_command_delay_ticks"] == 0
    assert nominal["use_gear_autobox"] is True
    assert abs(nominal["gear_switch_time_s"]) < 1e-12

    rng_a = random.Random(12345)
    rng_b = random.Random(12345)

    a = [
        sample_domain_randomization(rng=rng_a)
        for _ in range(1000)
    ]
    b = [
        sample_domain_randomization(rng=rng_b)
        for _ in range(1000)
    ]

    assert a == b, "Sampler không deterministic với cùng seed."

    one_tick_count = 0

    for sample in a:
        for key in (
            "torque_scale",
            "max_brake_torque",
            "steer_gain_positive",
            "steer_gain_negative",
        ):
            assert in_range(
                sample[key],
                DOMAIN_RANDOMIZATION_RANGES[key],
            ), "{} out of range: {}".format(
                key,
                sample[key],
            )

        assert sample["extra_command_delay_ticks"] in (0, 1)
        one_tick_count += sample["extra_command_delay_ticks"]

        assert abs(sample["control_hz"] - 50.0) < 1e-12
        assert abs(sample["control_dt_s"] - 0.020) < 1e-12
        assert abs(sample["camera_fps"] - 30.0) < 1e-12
        assert sample["use_gear_autobox"] is True
        assert abs(sample["gear_switch_time_s"]) < 1e-12

    one_tick_fraction = one_tick_count / float(len(a))

    print("Samples checked :", len(a))
    print(
        "1-tick delay     : {:.1%} (expected around 25%)".format(
            one_tick_fraction
        )
    )

    print("")
    print("Nominal:")
    print(nominal)

    print("")
    print("Example randomized sample:")
    print(a[0])

    print("")
    print("RESULT: GĐ6 DOMAIN RANDOMIZATION CONFIG PASS")
    print("=" * 88)


if __name__ == "__main__":
    main()
