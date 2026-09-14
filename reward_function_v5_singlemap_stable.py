# -*- coding: utf-8 -*-
"""
Reward V5 SINGLE-MAP STABLE 0.35 m/s
=====================================

Purpose
-------
Keep PPO action=[steer, speed] and obs100 unchanged, but make the task simple
and bounded for a real 1/10-scale vehicle.

The speed reward follows the ORIGINAL repository shape exactly, expressed as a
fraction of MAX speed:

    MIN    = MAX * 15/25
    TARGET = MAX * 22/25
    MAX    = configured maximum speed

With MAX=0.35 m/s:
    MIN    = 0.210 m/s
    TARGET = 0.308 m/s
    MAX    = 0.350 m/s

Main non-terminal reward:
    speed_factor * centering_factor * heading_factor

No recovery reward.
No adaptive speed by recovery state.
No progress bonus.
No command-target penalty.
No artificial "turn harder" bonus.

Recovery state remains diagnostic only so existing tooling can inspect how far
the car is from the lane center / heading.

Python 3.7 compatible.
"""
from __future__ import print_function

import math
from dataclasses import dataclass


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


@dataclass
class RewardConfigV5:
    # Original project ratios: min=15 km/h, target=22 km/h, max=25 km/h.
    original_min_to_max_ratio: float = 15.0 / 25.0
    original_target_to_max_ratio: float = 22.0 / 25.0

    # Miniature-track shaping (real-equivalent metres/radians).
    max_center_error_m: float = 0.18
    heading_zero_reward_rad: float = math.radians(45.0)

    # Reward-only pure-pursuit style preview. PPO observation is unchanged.
    # 0.50 m ~= 1.9 wheelbases for the 0.258 m real vehicle.
    heading_lookahead_m: float = 0.50

    # TRAIN-ONLY privileged oracle steering shaping.
    # PPO observation is unchanged; waypoint/oracle data never enters obs100.
    teacher_steer_weight: float = 0.50
    teacher_steer_error_scale: float = 0.35

    collision_penalty: float = 10.0
    offroad_penalty: float = 10.0
    stuck_penalty: float = 10.0
    overspeed_penalty: float = 10.0

    # Diagnostic-only state thresholds.
    adaptive_lateral_full_scale_m: float = 0.10
    adaptive_heading_full_scale_rad: float = math.radians(12.0)
    mild_severity: float = 0.25
    moderate_severity: float = 0.50
    strong_severity: float = 0.75
    recovery_success_hold_ticks: int = 8

    adaptive_speed_enabled: bool = False


class RewardV5(object):
    STATE_STABLE = "stable"
    STATE_MILD = "mild"
    STATE_MODERATE = "moderate"
    STATE_STRONG = "strong"

    def __init__(self, config=None):
        self.cfg = config if config is not None else RewardConfigV5()

    def compute_severity(self, lateral_error_m, heading_error_rad):
        lat_scale = max(float(self.cfg.adaptive_lateral_full_scale_m), 1e-9)
        head_scale = max(float(self.cfg.adaptive_heading_full_scale_rad), 1e-9)
        lat_severity = _clip(abs(float(lateral_error_m)) / lat_scale, 0.0, 1.0)
        heading_severity = _clip(abs(float(heading_error_rad)) / head_scale, 0.0, 1.0)
        severity = max(lat_severity, heading_severity)
        return float(severity), float(lat_severity), float(heading_severity)

    def classify_recovery_state(self, severity):
        severity = _clip(severity, 0.0, 1.0)
        if severity >= float(self.cfg.strong_severity):
            return self.STATE_STRONG
        if severity >= float(self.cfg.moderate_severity):
            return self.STATE_MODERATE
        if severity >= float(self.cfg.mild_severity):
            return self.STATE_MILD
        return self.STATE_STABLE

    def speed_limits(self, max_speed_mps):
        max_speed = max(1e-6, float(max_speed_mps))
        min_speed = max_speed * float(self.cfg.original_min_to_max_ratio)
        target_speed = max_speed * float(self.cfg.original_target_to_max_ratio)
        return float(min_speed), float(target_speed), float(max_speed)

    def target_speed_for_state(self, cruise_target_speed_mps, recovery_state):
        # Here cruise_target_speed_mps is intentionally the configured HARD MAX.
        # All recovery states share one original-style target.
        return self.speed_limits(cruise_target_speed_mps)[1]

    def state_and_target(self, cruise_target_speed_mps, lateral_error_m, heading_error_rad):
        severity, lat_severity, heading_severity = self.compute_severity(
            lateral_error_m=lateral_error_m,
            heading_error_rad=heading_error_rad,
        )
        state = self.classify_recovery_state(severity)
        target = self.target_speed_for_state(cruise_target_speed_mps, state)
        return state, float(target), severity, lat_severity, heading_severity

    def speed_factor(self, speed_mps, max_speed_mps):
        speed = max(0.0, float(speed_mps))
        min_speed, target_speed, max_speed = self.speed_limits(max_speed_mps)

        if speed < min_speed:
            factor = speed / max(min_speed, 1e-9)
        elif speed <= target_speed:
            factor = 1.0
        elif speed < max_speed:
            factor = 1.0 - (
                (speed - target_speed)
                / max(max_speed - target_speed, 1e-9)
            )
        else:
            factor = 0.0

        return _clip(factor, 0.0, 1.0)

    def compute(
        self,
        progress_delta_m,
        lateral_error_m,
        heading_error_rad,
        prev_lateral_error_m,
        prev_heading_error_rad,
        speed_mps,
        cruise_target_speed_mps,
        steer_cmd,
        prev_steer_cmd,
        speed_cmd_mps,
        prev_speed_cmd_mps,
        dt,
        teacher_steer_target_unit=0.0,
        recovery_success_event=False,
        collision=False,
        offroad=False,
        stuck=False,
        overspeed=False,
    ):
        lat = abs(float(lateral_error_m))
        heading = abs(float(heading_error_rad))
        speed = max(0.0, float(speed_mps))
        max_speed = max(1e-6, float(cruise_target_speed_mps))

        state, target_speed, severity, lat_sev, head_sev = self.state_and_target(
            max_speed, lat, heading
        )

        speed_factor = self.speed_factor(speed, max_speed)
        center_factor = _clip(
            1.0 - lat / max(float(self.cfg.max_center_error_m), 1e-9),
            0.0,
            1.0,
        )
        heading_factor = _clip(
            1.0 - heading / max(float(self.cfg.heading_zero_reward_rad), 1e-9),
            0.0,
            1.0,
        )

        teacher_target = _clip(
            float(teacher_steer_target_unit),
            -1.0,
            1.0,
        )
        teacher_error = abs(
            float(steer_cmd)
            - float(teacher_target)
        )
        teacher_factor = _clip(
            1.0
            - teacher_error
            / max(
                float(
                    self.cfg.teacher_steer_error_scale
                ),
                1e-9,
            ),
            0.0,
            1.0,
        )
        teacher_weight = _clip(
            float(
                self.cfg.teacher_steer_weight
            ),
            0.0,
            1.0,
        )

        # Keep original lane/speed objective, but replace the zero-gradient
        # corner plateau with a direct train-only steering teacher signal.
        direction_factor = (
            (1.0 - teacher_weight)
            * float(heading_factor)
            + teacher_weight
            * float(teacher_factor)
        )

        base_reward = float(
            speed_factor
            * center_factor
            * direction_factor
        )

        if collision:
            reward = -float(self.cfg.collision_penalty)
        elif offroad:
            reward = -float(self.cfg.offroad_penalty)
        elif stuck:
            reward = -float(self.cfg.stuck_penalty)
        elif overspeed:
            reward = -float(self.cfg.overspeed_penalty)
        else:
            reward = base_reward

        min_speed, target_speed, max_speed = self.speed_limits(max_speed)

        # Compatibility keys expected by the V5 diagnostic stack.
        terms = {
            "step": 0.0,
            "progress": 0.0,
            "lane": 0.0,
            "heading": 0.0,
            "speed_score": float(base_reward if reward >= 0.0 else 0.0),
            "speed_cmd_target": 0.0,
            "measured_speed_target": 0.0,
            "recovery_lateral_progress": 0.0,
            "recovery_heading_progress": 0.0,
            "recovery_success_bonus": 0.0,
            "steer_smooth": 0.0,
            "speed_cmd_smooth": 0.0,
            "collision": -float(self.cfg.collision_penalty) if collision else 0.0,
            "offroad": -float(self.cfg.offroad_penalty) if offroad else 0.0,
            "stuck": -float(self.cfg.stuck_penalty) if stuck else 0.0,
            "overspeed": -float(self.cfg.overspeed_penalty) if overspeed else 0.0,
        }

        terms["_final_reward"] = float(reward)
        terms["_simple_base_reward"] = float(base_reward)
        terms["_speed_factor"] = float(speed_factor)
        terms["_centering_factor"] = float(center_factor)
        terms["_heading_factor"] = float(heading_factor)
        terms["_teacher_steer_target"] = float(
            teacher_target
        )
        terms["_teacher_steer_error"] = float(
            teacher_error
        )
        terms["_teacher_steer_factor"] = float(
            teacher_factor
        )
        terms["_teacher_steer_weight"] = float(
            teacher_weight
        )
        terms["_direction_factor"] = float(
            direction_factor
        )
        terms["_min_speed_mps"] = float(min_speed)
        terms["_target_speed_mps"] = float(target_speed)
        terms["_max_speed_mps"] = float(max_speed)

        terms["_recovery_state"] = str(state)
        terms["_recovery_active"] = bool(state != self.STATE_STABLE)
        terms["_cruise_target_speed_mps"] = float(max_speed)
        terms["_adaptive_target_speed_mps"] = float(target_speed)
        terms["_adaptive_speed_severity"] = float(severity)
        terms["_adaptive_lat_severity"] = float(lat_sev)
        terms["_adaptive_heading_severity"] = float(head_sev)
        terms["_speed_cmd_error_mps"] = float(speed_cmd_mps) - float(target_speed)
        terms["_measured_speed_error_mps"] = float(speed) - float(target_speed)
        terms["_overspeed_multiplier"] = 1.0
        terms["_cmd_penalty_multiplier"] = 1.0
        terms["_measured_penalty_multiplier"] = 1.0
        terms["_lane_error_outside_corridor_m"] = float(lat)
        terms["_lane_safe_corridor_m"] = 0.0
        terms["_raw_progress_m"] = float(progress_delta_m)
        terms["_progress_for_reward_m"] = 0.0
        terms["_lateral_progress_m"] = 0.0
        terms["_heading_progress_rad"] = 0.0
        terms["_adaptive_speed_enabled"] = False

        return float(reward), terms
