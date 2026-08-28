# -*- coding: utf-8 -*-
"""
GĐ12 Reward V3 - anti-stop version.

Privileged signals are used only for reward/termination in simulation.
They are NOT part of the PPO observation.

Anti-stop rule:
- Every step has a small negative cost.
- Positive lane/heading/speed shaping is gated by measured vehicle motion.
- At speed == 0, lane/heading/speed shaping == 0.
"""

import math
from dataclasses import dataclass


def _clip(value, low, high):
    return max(low, min(high, float(value)))


@dataclass
class RewardConfigV3:
    # Anti-stop: standing still should not produce positive return.
    step_penalty: float = 0.005
    motion_gate_full_speed_mps: float = 0.20

    # Positive shaping.
    progress_weight: float = 3.0
    lane_score_weight: float = 0.030
    heading_score_weight: float = 0.020
    speed_score_weight: float = 0.020

    # Smoothness penalties.
    steer_delta_weight: float = 0.030
    speed_cmd_delta_weight: float = 0.020

    # Event penalties.
    collision_penalty: float = 5.0
    offroad_penalty: float = 3.0
    stuck_penalty: float = 2.0

    # Gaussian shaping widths in REAL-EQUIVALENT units.
    lane_sigma_m: float = 0.20
    heading_sigma_rad: float = math.radians(20.0)
    speed_sigma_mps: float = 0.25

    # Guard against abnormal progress spikes.
    max_progress_per_step_m: float = 0.20


class RewardV3:
    def __init__(self, config=None):
        self.cfg = config if config is not None else RewardConfigV3()

    @staticmethod
    def _gaussian_score(error, sigma):
        sigma = max(float(sigma), 1e-9)
        x = float(error) / sigma
        return math.exp(-(x * x))

    def compute(
        self,
        progress_delta_m,
        lateral_error_m,
        heading_error_rad,
        speed_mps,
        target_speed_mps,
        steer_cmd,
        prev_steer_cmd,
        speed_cmd_mps,
        prev_speed_cmd_mps,
        collision=False,
        offroad=False,
        stuck=False,
    ):
        progress_delta_m = _clip(
            progress_delta_m,
            -self.cfg.max_progress_per_step_m,
            +self.cfg.max_progress_per_step_m,
        )

        speed_mps = max(float(speed_mps), 0.0)

        lateral_error_m = abs(float(lateral_error_m))
        heading_error_rad = abs(float(heading_error_rad))
        speed_error_mps = abs(
            speed_mps - float(target_speed_mps)
        )

        steer_delta = (
            float(steer_cmd) - float(prev_steer_cmd)
        )
        speed_cmd_delta = (
            float(speed_cmd_mps) - float(prev_speed_cmd_mps)
        )

        # 0 m/s -> no positive shaping.
        # >= 0.20 m/s -> full shaping.
        motion_gate = _clip(
            speed_mps / max(
                self.cfg.motion_gate_full_speed_mps,
                1e-9,
            ),
            0.0,
            1.0,
        )

        lane_score = self._gaussian_score(
            lateral_error_m,
            self.cfg.lane_sigma_m,
        )
        heading_score = self._gaussian_score(
            heading_error_rad,
            self.cfg.heading_sigma_rad,
        )
        speed_score = self._gaussian_score(
            speed_error_mps,
            self.cfg.speed_sigma_mps,
        )

        terms = {
            "step": -self.cfg.step_penalty,

            "progress": (
                self.cfg.progress_weight
                * progress_delta_m
            ),

            "lane": (
                self.cfg.lane_score_weight
                * lane_score
                * motion_gate
            ),

            "heading": (
                self.cfg.heading_score_weight
                * heading_score
                * motion_gate
            ),

            "speed": (
                self.cfg.speed_score_weight
                * speed_score
                * motion_gate
            ),

            "steer_smooth": (
                -self.cfg.steer_delta_weight
                * steer_delta * steer_delta
            ),

            "speed_cmd_smooth": (
                -self.cfg.speed_cmd_delta_weight
                * speed_cmd_delta * speed_cmd_delta
            ),

            "collision": (
                -self.cfg.collision_penalty
                if collision else 0.0
            ),

            "offroad": (
                -self.cfg.offroad_penalty
                if offroad else 0.0
            ),

            "stuck": (
                -self.cfg.stuck_penalty
                if stuck else 0.0
            ),
        }

        reward = float(sum(terms.values()))

        # Diagnostic only; underscore key is NOT part of reward sum above.
        terms["_motion_gate"] = float(motion_gate)

        return reward, terms
