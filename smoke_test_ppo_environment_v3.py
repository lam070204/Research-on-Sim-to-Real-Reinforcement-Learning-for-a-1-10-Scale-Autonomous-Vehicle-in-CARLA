# -*- coding: utf-8 -*-
"""
GĐ14.2 - PPO <-> clean CARLA environment integration smoke test.

Tests:
- PPO receives obs(100)
- PPO sampled physical action respects:
      steer [-1,+1]
      speed [0,1]
- Environment accepts PPO action directly
- action -> observation prev-command mapping remains correct
- time_limit is truncated, not terminated
- PPO records V(next_obs) bootstrap for the truncation
- PPO learn() runs after the CARLA rollout

Run:
    python .\test_ppo_environment_v3.py
"""

import math
import numpy as np
import torch

from simulation.carla_connection_v3 import carla
from simulation.carla_environment_rgb_v3 import CarlaEnvironmentRGB
from networks.on_policy.ppo.ppo_agent_v3 import PPOAgent

from observation_builder_rgb_v3 import (
    OBSERVATION_DIM_V3,
    IDX_PREV_STEER,
    IDX_PREV_SPEED,
)


DESIRED_SPEED_MPS = 0.60
EPISODE_SECONDS = 1.5


def make_safe_initial_policy(agent):
    """
    Test-only initialization:
    raw mean [0, atanh(0.2)] -> physical action [0 steer, 0.6 m/s].
    Keeps the untrained smoke test on-road while still sampling PPO actions.
    """
    raw_speed_mean = math.atanh(
        2.0 * DESIRED_SPEED_MPS - 1.0
    )

    with torch.no_grad():
        final_linear = agent.policy.actor[-1]
        final_linear.weight.zero_()
        final_linear.bias.zero_()
        final_linear.bias[0] = 0.0
        final_linear.bias[1] = float(
            raw_speed_mean
        )

    agent.old_policy.load_state_dict(
        agent.policy.state_dict()
    )


def main():
    print("=" * 112)
    print("GĐ14.2 - PPO <-> ENVIRONMENT V3 INTEGRATION TEST")
    print("=" * 112)

    torch.manual_seed(0)
    np.random.seed(0)

    client = carla.Client(
        "localhost",
        2000,
    )
    client.set_timeout(30.0)

    world = client.get_world()

    env = None

    try:
        env = CarlaEnvironmentRGB(
            client=client,
            world=world,
            safe_spawn_numbers=None,
            desired_speed_mps=
                DESIRED_SPEED_MPS,
            max_episode_seconds=
                EPISODE_SECONDS,
            encoder_device="cpu",
        )

        agent = PPOAgent(
            town="_gd14_env_test",
            action_std_init=0.02,
            device="cpu",
        )

        make_safe_initial_policy(
            agent
        )

        obs = env.reset()

        if tuple(obs.shape) != (
            OBSERVATION_DIM_V3,
        ):
            raise RuntimeError(
                "RESET obs shape sai: {}".format(
                    tuple(obs.shape)
                )
            )

        done = False
        final_info = None
        tick = 0

        while not done and tick < 90:
            action = agent.get_action(
                obs,
                train=True,
            )

            next_obs, reward, done, info = (
                env.step(action)
            )

            agent.record_outcome(
                reward=reward,
                terminated=info[
                    "terminated"
                ],
                truncated=info[
                    "truncated"
                ],
                next_obs=(
                    next_obs
                    if info["truncated"]
                    else None
                ),
            )

            actual_steer = info[
                "control"
            ][
                "actual_steer_cmd"
            ]

            actual_speed = info[
                "control"
            ][
                "actual_speed_cmd_mps"
            ]

            obs_prev_steer = float(
                next_obs[
                    IDX_PREV_STEER
                ]
            )

            obs_prev_speed = float(
                next_obs[
                    IDX_PREV_SPEED
                ]
            )

            if abs(
                obs_prev_steer
                - actual_steer
            ) > 1e-5:
                raise RuntimeError(
                    "prev steer mismatch."
                )

            if abs(
                obs_prev_speed
                - actual_speed
            ) > 1e-5:
                raise RuntimeError(
                    "prev speed mismatch."
                )

            if tick % 10 == 0 or done:
                print(
                    "tick={:03d} | "
                    "action=({:+.3f},{:.3f}) | "
                    "v={:.3f} | "
                    "r={:+.4f} | "
                    "done={} term={} trunc={} "
                    "reason={}".format(
                        tick,
                        float(action[0]),
                        float(action[1]),
                        info["imu"][
                            "speed_mps"
                        ],
                        reward,
                        done,
                        info[
                            "terminated"
                        ],
                        info[
                            "truncated"
                        ],
                        info[
                            "termination_reason"
                        ],
                    )
                )

            obs = next_obs
            final_info = info
            tick += 1

        if not done:
            raise RuntimeError(
                "FAIL: episode không kết thúc."
            )

        if final_info is None:
            raise RuntimeError(
                "FAIL: không có final info."
            )

        if final_info[
            "termination_reason"
        ] != "time_limit":
            raise RuntimeError(
                "FAIL: expected time_limit, got {}".format(
                    final_info[
                        "termination_reason"
                    ]
                )
            )

        if final_info[
            "terminated"
        ]:
            raise RuntimeError(
                "FAIL: time_limit không được terminated=True."
            )

        if not final_info[
            "truncated"
        ]:
            raise RuntimeError(
                "FAIL: time_limit phải truncated=True."
            )

        if len(
            agent.memory.rewards
        ) != tick:
            raise RuntimeError(
                "FAIL: PPO rollout length mismatch."
            )

        bootstrap = float(
            agent.memory
            .bootstrap_values[-1]
        )

        if not np.isfinite(
            bootstrap
        ):
            raise RuntimeError(
                "FAIL: truncation bootstrap NaN/Inf."
            )

        print(
            "time-limit bootstrap V(next_obs) = {:+.6f}".format(
                bootstrap
            )
        )

        # This rollout ends at a boundary, so no last_obs argument is needed.
        metrics = agent.learn()

        if len(
            agent.memory.rewards
        ) != 0:
            raise RuntimeError(
                "FAIL: PPO memory chưa clear sau learn."
            )

        print(
            "PPO learn metrics:",
            metrics,
        )

        print("\n" + "=" * 112)
        print(
            "RESULT: GĐ14.2 PPO-ENV INTEGRATION PASS"
        )
        print("=" * 112)

    finally:
        if env is not None:
            env.close()

        print("Đã cleanup.")


if __name__ == "__main__":
    main()
