# -*- coding: utf-8 -*-
# GĐ6 CARLA environment integration smoke test.
#
# Run:
#   python .\smoke_test_gd6_environment_v3.py

from __future__ import print_function

import math
import numpy as np

from simulation.carla_connection_v3 import carla
from simulation.carla_environment_rgb_v3 import CarlaEnvironmentRGB
from domain_randomization_v3 import DOMAIN_RANDOMIZATION_RANGES


def in_range(value, pair):
    return (
        float(pair[0])
        <= float(value)
        <= float(pair[1])
    )


def main():
    print("=" * 100)
    print("GĐ6 - CARLA ENVIRONMENT DOMAIN RANDOMIZATION SMOKE TEST")
    print("=" * 100)

    client = carla.Client("localhost", 2000)
    client.set_timeout(120.0)
    world = client.get_world()

    env = None

    try:
        env = CarlaEnvironmentRGB(
            client=client,
            world=world,
            safe_spawn_numbers=[2],
            desired_speed_mps=0.40,
            max_episode_seconds=1.0,
            encoder_device="cpu",
            domain_randomization_enabled=True,
            domain_randomization_seed=6106,
        )

        seen = []

        for episode in range(3):
            obs = env.reset()
            assert tuple(obs.shape) == (100,), tuple(obs.shape)

            params = dict(env.domain_randomization_params)

            for key in (
                "torque_scale",
                "max_brake_torque",
                "steer_gain_positive",
                "steer_gain_negative",
            ):
                assert in_range(
                    params[key],
                    DOMAIN_RANDOMIZATION_RANGES[key],
                ), (key, params[key])

            assert params["extra_command_delay_ticks"] in (0, 1)

            action = np.asarray(
                [0.20, 0.30],
                dtype=np.float32,
            )

            next_obs, reward, done, info = env.step(action)

            assert tuple(next_obs.shape) == (100,)
            assert math.isfinite(float(reward))
            assert "domain_randomization" in info
            assert "control" in info

            control = info["control"]

            assert abs(
                float(env.action_controller.prev_steer_cmd)
                - float(control["actual_steer_cmd"])
            ) < 1e-6

            assert abs(
                float(env.action_controller.prev_speed_cmd_mps)
                - float(control["actual_speed_cmd_mps"])
            ) < 1e-6

            seen.append(params)

            print(
                "EP{:02d} PASS | torque={:.4f} | brake={:.1f} | "
                "gain(+/-)={:.4f}/{:.4f} | delay={} | "
                "logical=({:+.3f},{:.3f}) | physical_steer={:+.3f}".format(
                    episode + 1,
                    float(params["torque_scale"]),
                    float(params["max_brake_torque"]),
                    float(params["steer_gain_positive"]),
                    float(params["steer_gain_negative"]),
                    int(params["extra_command_delay_ticks"]),
                    float(control["actual_steer_cmd"]),
                    float(control["actual_speed_cmd_mps"]),
                    float(control["gd6_physical_steer_cmd"]),
                )
            )

        torque_values = [
            round(float(x["torque_scale"]), 8)
            for x in seen
        ]
        assert len(set(torque_values)) > 1, torque_values

        print("")
        print("RESULT: GĐ6 CARLA ENVIRONMENT INTEGRATION PASS")
        print("=" * 100)

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
