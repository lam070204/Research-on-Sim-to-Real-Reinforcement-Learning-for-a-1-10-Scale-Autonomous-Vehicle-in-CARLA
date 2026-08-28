# -*- coding: utf-8 -*-
"""
GĐ12/GĐ13 canonical reward + termination manager.

Important:
- `speed_cmd_mps` is the actual policy command after clamp/rate-limit.
  It is used for STUCK detection and smoothness.
- `desired_speed_mps` is the task/curriculum speed reference.
  It is used by the speed reward.

This separation prevents the policy from choosing an artificially easy
speed target for its own reward.

Backward compatibility:
- If desired_speed_mps is omitted, it falls back to speed_cmd_mps.
"""

from dataclasses import dataclass

from reward_function_v3 import RewardV3, RewardConfigV3
from reward_metrics_carla_v3 import (
    RoadRewardMetricsV3,
    CollisionTrackerV3,
)


@dataclass
class EpisodeConfigV3:
    max_episode_seconds: float = 60.0
    terminate_on_collision: bool = True
    terminate_on_offroad: bool = True
    terminate_on_stuck: bool = True


class RewardManagerV3:
    def __init__(
        self,
        vehicle,
        reward_config=None,
        episode_config=None,
    ):
        self.vehicle = vehicle

        self.reward_fn = RewardV3(
            config=(
                reward_config
                if reward_config is not None
                else RewardConfigV3()
            )
        )

        self.episode_cfg = (
            episode_config
            if episode_config is not None
            else EpisodeConfigV3()
        )

        self.road_metrics = RoadRewardMetricsV3(vehicle)
        self.collision_tracker = CollisionTrackerV3(vehicle)

        self.elapsed_s = 0.0
        self.step_count = 0
        self.episode_reward = 0.0

        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0

        self.reset()

    def reset(
        self,
        prev_steer_cmd=0.0,
        prev_speed_cmd_mps=0.0,
    ):
        self.elapsed_s = 0.0
        self.step_count = 0
        self.episode_reward = 0.0

        self.prev_steer_cmd = float(prev_steer_cmd)
        self.prev_speed_cmd_mps = float(prev_speed_cmd_mps)

        self.road_metrics.reset()
        self.collision_tracker.reset()

    def _termination(self, collision, offroad, stuck):
        if (
            self.episode_cfg.terminate_on_collision
            and collision
        ):
            return True, False, "collision"

        if (
            self.episode_cfg.terminate_on_offroad
            and offroad
        ):
            return True, False, "offroad"

        if (
            self.episode_cfg.terminate_on_stuck
            and stuck
        ):
            return True, False, "stuck"

        if (
            self.elapsed_s
            >= self.episode_cfg.max_episode_seconds
        ):
            # Time limit is a truncation, not a physical terminal event.
            return False, True, "time_limit"

        return False, False, None

    def step(
        self,
        speed_mps,
        steer_cmd,
        speed_cmd_mps,
        dt,
        desired_speed_mps=None,
    ):
        """
        Call after ActionControllerV3.step() and after world.tick().

        Returns:
            reward, done, info
        """
        dt = max(float(dt), 0.0)

        speed_mps = float(speed_mps)
        steer_cmd = float(steer_cmd)
        speed_cmd_mps = float(speed_cmd_mps)

        if desired_speed_mps is None:
            desired_speed_mps = speed_cmd_mps
        desired_speed_mps = float(desired_speed_mps)

        self.elapsed_s += dt
        self.step_count += 1

        collision = self.collision_tracker.consume_collision()

        metrics = self.road_metrics.update(
            speed_mps=speed_mps,
            # Stuck asks: "did the agent command movement?"
            target_speed_cmd_mps=speed_cmd_mps,
            dt=dt,
            collision=collision,
        )

        reward, terms = self.reward_fn.compute(
            progress_delta_m=metrics["forward_progress_m"],
            lateral_error_m=metrics["lateral_error_m"],
            heading_error_rad=metrics["heading_error_rad"],
            speed_mps=speed_mps,
            # Speed shaping uses task reference, not the policy's own command.
            target_speed_mps=desired_speed_mps,
            steer_cmd=steer_cmd,
            prev_steer_cmd=self.prev_steer_cmd,
            speed_cmd_mps=speed_cmd_mps,
            prev_speed_cmd_mps=self.prev_speed_cmd_mps,
            collision=metrics["collision"],
            offroad=metrics["offroad"],
            stuck=metrics["stuck"],
        )

        terminated, truncated, reason = self._termination(
            collision=metrics["collision"],
            offroad=metrics["offroad"],
            stuck=metrics["stuck"],
        )
        done = bool(terminated or truncated)

        self.prev_steer_cmd = steer_cmd
        self.prev_speed_cmd_mps = speed_cmd_mps
        self.episode_reward += float(reward)

        info = {
            "reward_terms": terms,
            "desired_speed_mps": desired_speed_mps,

            "lateral_error_m": metrics["lateral_error_m"],
            "heading_error_rad": metrics["heading_error_rad"],
            "forward_progress_m": metrics["forward_progress_m"],

            "collision": metrics["collision"],
            "offroad": metrics["offroad"],
            "stuck": metrics["stuck"],
            "stuck_time_s": metrics["stuck_time_s"],

            "episode_time_s": float(self.elapsed_s),
            "episode_steps": int(self.step_count),
            "episode_reward": float(self.episode_reward),

            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": done,
            "termination_reason": reason,
        }

        return float(reward), done, info

    def destroy(self):
        if self.collision_tracker is not None:
            self.collision_tracker.destroy()
            self.collision_tracker = None
