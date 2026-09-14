# -*- coding: utf-8 -*-
"""
Reward V5 SIMPLE - original-repo style.

Keeps PPO action=[steer, speed] and obs100 unchanged, but deliberately removes
state-dependent speed targets and recovery shaping from the reward.

Main reward (when no terminal event):
    speed_factor * centering_factor * heading_factor

Speed factor follows the original repository logic:
    speed < min_speed       -> speed/min_speed
    min_speed..target_speed -> 1.0
    target..max_speed       -> linearly falls from 1.0 to 0.0

Recovery state is diagnostic only. It does NOT change target speed or reward.
Python 3.7 compatible.
"""
from __future__ import print_function

import math
from dataclasses import dataclass


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


@dataclass
class RewardConfigV5:
    # Ratios copied from the original project: min=15, target=22, max=25 km/h.
    # With target_speed=0.50 m/s this becomes ~0.341 / 0.500 / 0.568 m/s.
    original_min_to_target_ratio: float = 15.0 / 22.0
    original_max_to_target_ratio: float = 25.0 / 22.0

    # Spatial/heading shaping for the miniature track (real-equivalent metres).
    # Like the original code, reward falls linearly to zero at this limit.
    max_center_error_m: float = 0.10
    heading_zero_reward_rad: float = math.radians(20.0)

    # Terminal penalties follow the original repository's simple -10 style.
    collision_penalty: float = 10.0
    offroad_penalty: float = 10.0
    stuck_penalty: float = 10.0
    overspeed_penalty: float = 10.0

    # Diagnostic-only recovery state thresholds. These never alter reward/speed.
    adaptive_lateral_full_scale_m: float = 0.10
    adaptive_heading_full_scale_rad: float = math.radians(12.0)
    mild_severity: float = 0.25
    moderate_severity: float = 0.50
    strong_severity: float = 0.75
    recovery_success_hold_ticks: int = 8

    # Compatibility flag for existing trainer diagnostics.
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

    def target_speed_for_state(self, cruise_target_speed_mps, recovery_state):
        # SIMPLE V5: one target speed for every state. State is diagnostic only.
        return max(0.0, float(cruise_target_speed_mps))

    def state_and_target(self, cruise_target_speed_mps, lateral_error_m, heading_error_rad):
        severity, lat_severity, heading_severity = self.compute_severity(
            lateral_error_m=lateral_error_m,
            heading_error_rad=heading_error_rad,
        )
        state = self.classify_recovery_state(severity)
        target = self.target_speed_for_state(cruise_target_speed_mps, state)
        return state, float(target), severity, lat_severity, heading_severity

    def speed_limits(self, target_speed_mps):
        target = max(1e-6, float(target_speed_mps))
        min_speed = target * float(self.cfg.original_min_to_target_ratio)
        max_speed = target * float(self.cfg.original_max_to_target_ratio)
        return float(min_speed), float(target), float(max_speed)

    def speed_factor(self, speed_mps, target_speed_mps):
        speed = max(0.0, float(speed_mps))
        min_speed, target, max_speed = self.speed_limits(target_speed_mps)

        # Exact shape used by the original project, rescaled to m/s.
        if speed < min_speed:
            factor = speed / max(min_speed, 1e-9)
        elif speed > target:
            factor = 1.0 - (speed - target) / max(max_speed - target, 1e-9)
        else:
            factor = 1.0
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
        recovery_success_event=False,
        collision=False,
        offroad=False,
        stuck=False,
        overspeed=False,
    ):
        # Deliberately ignore progress, action deltas and recovery bonuses.
        # They stay in the signature so RewardManager/trainer compatibility is kept.
        lat = abs(float(lateral_error_m))
        heading = abs(float(heading_error_rad))
        speed = max(0.0, float(speed_mps))
        target_speed = max(1e-6, float(cruise_target_speed_mps))

        state, _, severity, lat_sev, head_sev = self.state_and_target(
            target_speed, lat, heading
        )

        speed_factor = self.speed_factor(speed, target_speed)
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

        base_reward = float(speed_factor * center_factor * heading_factor)

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

        min_speed, target, max_speed = self.speed_limits(target_speed)

        # Keep the historical keys expected by V5 diagnostics, but only
        # "speed_score" carries the non-terminal SIMPLE reward. All removed
        # shaping terms are exactly zero.
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
        terms["_min_speed_mps"] = float(min_speed)
        terms["_target_speed_mps"] = float(target)
        terms["_max_speed_mps"] = float(max_speed)

        terms["_recovery_state"] = str(state)
        terms["_recovery_active"] = bool(state != self.STATE_STABLE)
        terms["_cruise_target_speed_mps"] = float(target_speed)
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
