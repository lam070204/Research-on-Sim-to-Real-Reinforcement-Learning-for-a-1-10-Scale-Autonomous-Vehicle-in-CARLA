# -*- coding: utf-8 -*-
"""
Reward V5 - CLEAN / STATE-AWARE SPEED REWARD

Thiáº¿t káº¿ cho PPO train láº¡i tá»« Ä‘áº§u.

Má»¥c tiÃªu chÃ­nh:
- Má»™t reward path duy nháº¥t.
- Recovery state Ä‘Æ°á»£c suy ra tá»« privileged lateral/heading error.
- Má»—i recovery state cÃ³ target speed rÃµ rÃ ng.
- PPO speed command Ä‘Æ°á»£c reward/penalty TRá»°C TIáº¾P theo target state.
- Forward progress KHÃ”NG Ä‘Æ°á»£c hÆ°á»Ÿng thÃªm lá»£i Ã­ch khi cháº¡y vÆ°á»£t target.
- Recovery progress Ä‘Æ°á»£c tÃ­nh ngay trong RewardV5, khÃ´ng cá»™ng delta á»Ÿ environment.
- Privileged state chá»‰ dÃ¹ng reward/termination, KHÃ”NG vÃ o PPO observation.

STABLE   -> 0.60 m/s
MILD     -> 0.50 m/s
MODERATE -> 0.40 m/s
STRONG   -> 0.30 m/s

Python 3.7 compatible.
"""

from __future__ import print_function

import math
from dataclasses import dataclass


def _clip(value, low, high):
    return max(float(low), min(float(high), float(value)))


@dataclass
class RewardConfigV5:
    # ------------------------------------------------------------------
    # Anti-stop / base shaping
    # ------------------------------------------------------------------
    step_penalty: float = 0.005
    motion_gate_full_speed_mps: float = 0.20

    # Sim-to-real safety margin: lane shaping is intentionally stronger
    # than the generic speed score, but uses a flat safe corridor so PPO
    # is NOT forced to chase the exact centerline millimeter-by-millimeter.
    lane_score_weight: float = 0.060
    heading_score_weight: float = 0.020
    speed_score_weight: float = 0.020

    lane_safe_corridor_m: float = 0.025
    lane_sigma_m: float = 0.045
    heading_sigma_rad: float = math.radians(20.0)
    speed_sigma_mps: float = 0.25

    # Forward progress.
    # Positive progress is capped by target_speed * dt so overspeed does not
    # earn extra progress reward. Negative progress is still penalized.
    progress_weight: float = 3.0
    progress_target_cap_factor: float = 1.10
    max_progress_per_step_m: float = 0.20

    # Smoothness.
    steer_delta_weight: float = 0.030
    speed_cmd_delta_weight: float = 0.020

    # ------------------------------------------------------------------
    # State-aware speed curriculum
    # ------------------------------------------------------------------
    adaptive_speed_enabled: bool = True

    # Severity normalization.
    adaptive_lateral_full_scale_m: float = 0.10
    adaptive_heading_full_scale_rad: float = math.radians(12.0)

    # Recovery state thresholds in normalized severity [0, 1].
    mild_severity: float = 0.25
    moderate_severity: float = 0.50
    strong_severity: float = 0.75

    # Target-speed fractions of cruise target.
    # With cruise=1.0 -> 0.60 / 0.50 / 0.40 / 0.30 m/s.
    stable_speed_fraction: float = 0.60
    mild_speed_fraction: float = 0.50
    moderate_speed_fraction: float = 0.40
    strong_speed_fraction: float = 0.30

    adaptive_min_target_speed_mps: float = 0.30
    adaptive_strong_target_cap_mps: float = 0.40

    # DIRECT action shaping.
    # Symmetric command tracking is intentional:
    # stable -> penalize staying too slow
    # recovery -> penalize asking too fast
    speed_cmd_target_weight: float = 0.30

    # Measured speed should also follow state target, but weaker than command.
    measured_speed_target_weight: float = 0.08

    # Extra OVERSPEED penalty by recovery severity.
    # Stable/Mild keep the original shaping.
    # Only positive speed error (speed > target) is amplified so recovery
    # is not punished extra for slowing down below the target.
    moderate_overspeed_multiplier: float = 2.0
    strong_overspeed_multiplier: float = 3.0

    # ------------------------------------------------------------------
    # Recovery progress reward
    # ------------------------------------------------------------------
    recovery_lateral_progress_weight: float = 2.50
    recovery_heading_progress_weight: float = 0.20
    recovery_success_bonus: float = 0.20

    max_lateral_progress_per_tick_m: float = 0.020
    max_heading_progress_per_tick_rad: float = math.radians(4.0)

    # A completed recovery requires STABLE for several ticks.
    recovery_success_hold_ticks: int = 8

    # ------------------------------------------------------------------
    # Event penalties
    # ------------------------------------------------------------------
    collision_penalty: float = 5.0
    offroad_penalty: float = 3.0
    stuck_penalty: float = 2.0


class RewardV5(object):
    STATE_STABLE = "stable"
    STATE_MILD = "mild"
    STATE_MODERATE = "moderate"
    STATE_STRONG = "strong"

    def __init__(self, config=None):
        self.cfg = config if config is not None else RewardConfigV5()

    @staticmethod
    def _gaussian_score(error, sigma):
        sigma = max(float(sigma), 1e-9)
        x = float(error) / sigma
        return math.exp(-(x * x))

    def compute_severity(
        self,
        lateral_error_m,
        heading_error_rad,
    ):
        lat_scale = max(
            float(self.cfg.adaptive_lateral_full_scale_m),
            1e-9,
        )
        heading_scale = max(
            float(self.cfg.adaptive_heading_full_scale_rad),
            1e-9,
        )

        lat_severity = _clip(
            abs(float(lateral_error_m)) / lat_scale,
            0.0,
            1.0,
        )
        heading_severity = _clip(
            abs(float(heading_error_rad)) / heading_scale,
            0.0,
            1.0,
        )

        severity = max(lat_severity, heading_severity)

        return (
            float(severity),
            float(lat_severity),
            float(heading_severity),
        )

    def classify_recovery_state(self, severity):
        severity = _clip(severity, 0.0, 1.0)

        if severity >= float(self.cfg.strong_severity):
            return self.STATE_STRONG

        if severity >= float(self.cfg.moderate_severity):
            return self.STATE_MODERATE

        if severity >= float(self.cfg.mild_severity):
            return self.STATE_MILD

        return self.STATE_STABLE

    def target_speed_for_state(
        self,
        cruise_target_speed_mps,
        recovery_state,
    ):
        cruise = max(float(cruise_target_speed_mps), 0.0)

        if not bool(self.cfg.adaptive_speed_enabled):
            return float(cruise)

        minimum = _clip(
            self.cfg.adaptive_min_target_speed_mps,
            0.0,
            cruise,
        )

        fractions = {
            self.STATE_STABLE: float(
                self.cfg.stable_speed_fraction
            ),
            self.STATE_MILD: float(
                self.cfg.mild_speed_fraction
            ),
            self.STATE_MODERATE: float(
                self.cfg.moderate_speed_fraction
            ),
            self.STATE_STRONG: float(
                self.cfg.strong_speed_fraction
            ),
        }

        fraction = fractions.get(
            str(recovery_state),
            float(self.cfg.stable_speed_fraction),
        )

        target = cruise * fraction

        if recovery_state == self.STATE_STRONG:
            target = min(
                target,
                float(self.cfg.adaptive_strong_target_cap_mps),
            )

        target = _clip(target, minimum, cruise)
        return float(target)

    def state_and_target(
        self,
        cruise_target_speed_mps,
        lateral_error_m,
        heading_error_rad,
    ):
        if not bool(self.cfg.adaptive_speed_enabled):
            return (
                self.STATE_STABLE,
                float(max(cruise_target_speed_mps, 0.0)),
                0.0,
                0.0,
                0.0,
            )

        (
            severity,
            lat_severity,
            heading_severity,
        ) = self.compute_severity(
            lateral_error_m=lateral_error_m,
            heading_error_rad=heading_error_rad,
        )

        state = self.classify_recovery_state(severity)

        target = self.target_speed_for_state(
            cruise_target_speed_mps=cruise_target_speed_mps,
            recovery_state=state,
        )

        return (
            state,
            float(target),
            float(severity),
            float(lat_severity),
            float(heading_severity),
        )

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
    ):
        dt = max(float(dt), 0.0)

        lateral_error_m = abs(float(lateral_error_m))
        heading_error_rad = abs(float(heading_error_rad))

        prev_lateral_error_m = abs(
            float(prev_lateral_error_m)
        )
        prev_heading_error_rad = abs(
            float(prev_heading_error_rad)
        )

        speed_mps = max(float(speed_mps), 0.0)
        speed_cmd_mps = max(float(speed_cmd_mps), 0.0)

        (
            recovery_state,
            target_speed_mps,
            severity,
            lat_severity,
            heading_severity,
        ) = self.state_and_target(
            cruise_target_speed_mps=cruise_target_speed_mps,
            lateral_error_m=lateral_error_m,
            heading_error_rad=heading_error_rad,
        )

        # ------------------------------------------------------------------
        # Forward progress
        # ------------------------------------------------------------------
        raw_progress = _clip(
            progress_delta_m,
            -self.cfg.max_progress_per_step_m,
            +self.cfg.max_progress_per_step_m,
        )

        if raw_progress >= 0.0:
            target_progress_cap = (
                float(target_speed_mps)
                * float(dt)
                * float(self.cfg.progress_target_cap_factor)
            )
            progress_for_reward = min(
                raw_progress,
                max(0.0, target_progress_cap),
            )
        else:
            progress_for_reward = raw_progress

        # ------------------------------------------------------------------
        # Base scores
        # ------------------------------------------------------------------
        motion_gate = _clip(
            speed_mps / max(
                float(self.cfg.motion_gate_full_speed_mps),
                1e-9,
            ),
            0.0,
            1.0,
        )

        # Flat high-reward corridor around centerline. Only the error OUTSIDE
        # this corridor is shaped. This discourages edge-hugging trajectories
        # without making the steering oscillate just to hit lat=0 exactly.
        lane_error_outside_corridor = max(
            0.0,
            float(lateral_error_m)
            - float(self.cfg.lane_safe_corridor_m),
        )
        lane_score = self._gaussian_score(
            lane_error_outside_corridor,
            self.cfg.lane_sigma_m,
        )
        heading_score = self._gaussian_score(
            heading_error_rad,
            self.cfg.heading_sigma_rad,
        )

        measured_speed_error = (
            speed_mps - float(target_speed_mps)
        )
        speed_score = self._gaussian_score(
            abs(measured_speed_error),
            self.cfg.speed_sigma_mps,
        )

        # ------------------------------------------------------------------
        # Direct state-speed shaping
        # ------------------------------------------------------------------
        cmd_speed_error = (
            speed_cmd_mps - float(target_speed_mps)
        )

        speed_cmd_target_penalty = 0.0
        measured_speed_target_penalty = 0.0

        # Keep the original symmetric tracking penalty as the base.
        # Add extra pressure ONLY when the vehicle asks for / reaches a speed
        # above the state target in MODERATE or STRONG recovery.
        overspeed_multiplier = 1.0

        if recovery_state == self.STATE_MODERATE:
            overspeed_multiplier = float(
                self.cfg.moderate_overspeed_multiplier
            )
        elif recovery_state == self.STATE_STRONG:
            overspeed_multiplier = float(
                self.cfg.strong_overspeed_multiplier
            )

        cmd_penalty_multiplier = (
            overspeed_multiplier
            if cmd_speed_error > 0.0
            else 1.0
        )
        measured_penalty_multiplier = (
            overspeed_multiplier
            if measured_speed_error > 0.0
            else 1.0
        )

        if bool(self.cfg.adaptive_speed_enabled):
            speed_cmd_target_penalty = (
                -float(self.cfg.speed_cmd_target_weight)
                * float(cmd_penalty_multiplier)
                * cmd_speed_error
                * cmd_speed_error
            )

            measured_speed_target_penalty = (
                -float(self.cfg.measured_speed_target_weight)
                * float(measured_penalty_multiplier)
                * measured_speed_error
                * measured_speed_error
            )

        # ------------------------------------------------------------------
        # Recovery progress
        # ------------------------------------------------------------------
        recovery_active = (
            recovery_state != self.STATE_STABLE
        )

        lateral_progress = 0.0
        heading_progress = 0.0

        if recovery_active:
            lateral_progress = _clip(
                prev_lateral_error_m - lateral_error_m,
                -float(
                    self.cfg.max_lateral_progress_per_tick_m
                ),
                +float(
                    self.cfg.max_lateral_progress_per_tick_m
                ),
            )

            heading_progress = _clip(
                prev_heading_error_rad - heading_error_rad,
                -float(
                    self.cfg.max_heading_progress_per_tick_rad
                ),
                +float(
                    self.cfg.max_heading_progress_per_tick_rad
                ),
            )

        recovery_lateral_reward = (
            float(self.cfg.recovery_lateral_progress_weight)
            * float(lateral_progress)
        )

        recovery_heading_reward = (
            float(self.cfg.recovery_heading_progress_weight)
            * float(heading_progress)
        )

        recovery_success_reward = (
            float(self.cfg.recovery_success_bonus)
            if recovery_success_event
            else 0.0
        )

        # ------------------------------------------------------------------
        # Smoothness
        # ------------------------------------------------------------------
        steer_delta = (
            float(steer_cmd) - float(prev_steer_cmd)
        )
        speed_cmd_delta = (
            float(speed_cmd_mps)
            - float(prev_speed_cmd_mps)
        )

        # ------------------------------------------------------------------
        # Reward terms
        # ------------------------------------------------------------------
        terms = {
            "step": -float(self.cfg.step_penalty),

            "progress": (
                float(self.cfg.progress_weight)
                * float(progress_for_reward)
            ),

            "lane": (
                float(self.cfg.lane_score_weight)
                * float(lane_score)
                * float(motion_gate)
            ),

            "heading": (
                float(self.cfg.heading_score_weight)
                * float(heading_score)
                * float(motion_gate)
            ),

            "speed_score": (
                float(self.cfg.speed_score_weight)
                * float(speed_score)
                * float(motion_gate)
            ),

            "speed_cmd_target": float(
                speed_cmd_target_penalty
            ),

            "measured_speed_target": float(
                measured_speed_target_penalty
            ),

            "recovery_lateral_progress": float(
                recovery_lateral_reward
            ),

            "recovery_heading_progress": float(
                recovery_heading_reward
            ),

            "recovery_success_bonus": float(
                recovery_success_reward
            ),

            "steer_smooth": (
                -float(self.cfg.steer_delta_weight)
                * steer_delta
                * steer_delta
            ),

            "speed_cmd_smooth": (
                -float(self.cfg.speed_cmd_delta_weight)
                * speed_cmd_delta
                * speed_cmd_delta
            ),

            "collision": (
                -float(self.cfg.collision_penalty)
                if collision else 0.0
            ),

            "offroad": (
                -float(self.cfg.offroad_penalty)
                if offroad else 0.0
            ),

            "stuck": (
                -float(self.cfg.stuck_penalty)
                if stuck else 0.0
            ),
        }

        reward = float(sum(terms.values()))

        # Diagnostics only. Keys beginning "_" are NOT summed above.
        terms["_final_reward"] = float(reward)

        terms["_recovery_state"] = str(recovery_state)
        terms["_recovery_active"] = bool(recovery_active)

        terms["_cruise_target_speed_mps"] = float(
            cruise_target_speed_mps
        )
        terms["_adaptive_target_speed_mps"] = float(
            target_speed_mps
        )

        terms["_adaptive_speed_severity"] = float(severity)
        terms["_adaptive_lat_severity"] = float(lat_severity)
        terms["_adaptive_heading_severity"] = float(
            heading_severity
        )

        terms["_speed_cmd_error_mps"] = float(
            cmd_speed_error
        )
        terms["_measured_speed_error_mps"] = float(
            measured_speed_error
        )

        terms["_overspeed_multiplier"] = float(
            overspeed_multiplier
        )
        terms["_cmd_penalty_multiplier"] = float(
            cmd_penalty_multiplier
        )
        terms["_measured_penalty_multiplier"] = float(
            measured_penalty_multiplier
        )

        terms["_lane_error_outside_corridor_m"] = float(
            lane_error_outside_corridor
        )
        terms["_lane_safe_corridor_m"] = float(
            self.cfg.lane_safe_corridor_m
        )

        terms["_raw_progress_m"] = float(raw_progress)
        terms["_progress_for_reward_m"] = float(
            progress_for_reward
        )

        terms["_lateral_progress_m"] = float(
            lateral_progress
        )
        terms["_heading_progress_rad"] = float(
            heading_progress
        )

        terms["_adaptive_speed_enabled"] = bool(
            self.cfg.adaptive_speed_enabled
        )

        return float(reward), terms