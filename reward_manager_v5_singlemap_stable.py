# -*- coding: utf-8 -*-
"""
Reward Manager V5 SINGLE-MAP STABLE - ORIGINAL-STYLE SPEED REWARD

Một reward path duy nhất:
    CARLA privileged metrics
        -> recovery state
        -> state target speed
        -> RewardV5
        -> FINAL reward
        -> PPO

Không inherit RewardManager V3/V4.
Không patch reward manager từ environment.
Không cộng recovery reward delta ở environment.

Python 3.7 compatible.
"""

from __future__ import print_function

import math

from dataclasses import dataclass

from reward_function_v5_singlemap_stable import (
    RewardV5,
    RewardConfigV5,
)
from reward_metrics_carla_v5 import (
    RoadRewardMetricsV5,
    CollisionTrackerV5,
)
from vehicle_specs_v5 import REAL


@dataclass
class EpisodeConfigV5:
    max_episode_seconds: float = 60.0
    terminate_on_collision: bool = True
    terminate_on_offroad: bool = True
    terminate_on_stuck: bool = True


class RewardManagerV5(object):
    def __init__(
        self,
        vehicle,
        reward_config=None,
        episode_config=None,
    ):
        self.vehicle = vehicle

        self.reward_fn = RewardV5(
            config=(
                reward_config
                if reward_config is not None
                else RewardConfigV5()
            )
        )

        self.episode_cfg = (
            episode_config
            if episode_config is not None
            else EpisodeConfigV5()
        )

        self.road_metrics = RoadRewardMetricsV5(
            vehicle
        )
        self.collision_tracker = (
            CollisionTrackerV5(
                vehicle
            )
        )

        self.elapsed_s = 0.0
        self.step_count = 0
        self.episode_reward = 0.0

        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0

        self.prev_abs_lateral_error_m = 0.0
        self.prev_abs_heading_error_rad = 0.0

        self.prev_recovery_state = (
            RewardV5.STATE_STABLE
        )
        self.recovery_active = False
        self.recovery_success_hold = 0
        self.recovery_event_count = 0
        self.recovery_success_count = 0

        self.reset()

    @staticmethod
    def _wrap_pi(angle_rad):
        return (
            (float(angle_rad) + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )

    def _preview_heading_error_rad(self, projected_wp):
        """
        Reward-only anticipatory heading error.

        Instead of comparing vehicle yaw only with the tangent directly under
        the vehicle, point toward a Driving waypoint ahead on the lane.  This
        is a pure-pursuit-like target bearing and gives a smooth signal before
        a 90-degree corner.

        PPO observation is NOT changed and receives no privileged waypoint.
        """
        current_error = float(
            self.road_metrics._heading_error_rad(projected_wp)
        )

        if projected_wp is None:
            return current_error, 0.0

        lookahead_real_m = max(
            0.0,
            float(
                getattr(
                    self.reward_fn.cfg,
                    "heading_lookahead_m",
                    0.0,
                )
            ),
        )

        if lookahead_real_m <= 1e-9:
            return current_error, 0.0

        lookahead_carla_m = (
            lookahead_real_m
            * float(self.road_metrics.carla_scale)
        )

        try:
            candidates = list(
                projected_wp.next(
                    float(lookahead_carla_m)
                )
            )
        except Exception:
            candidates = []

        if not candidates:
            return current_error, 0.0

        current_road_yaw = math.radians(
            float(
                projected_wp
                .transform
                .rotation
                .yaw
            )
        )

        # Prefer same lane when available.  If the OpenDRIVE road id changes
        # at a corner, fall back to the candidate with the smallest tangent
        # discontinuity.  This avoids arbitrary branch choice at a junction.
        same_lane = [
            wp
            for wp in candidates
            if (
                getattr(wp, "road_id", None)
                == getattr(projected_wp, "road_id", None)
                and
                getattr(wp, "lane_id", None)
                == getattr(projected_wp, "lane_id", None)
            )
        ]
        pool = same_lane if same_lane else candidates

        def candidate_score(wp):
            try:
                yaw = math.radians(
                    float(
                        wp.transform.rotation.yaw
                    )
                )
                return abs(
                    self._wrap_pi(
                        yaw - current_road_yaw
                    )
                )
            except Exception:
                return float("inf")

        target_wp = min(
            pool,
            key=candidate_score,
        )

        vehicle_location = (
            self.vehicle.get_location()
        )
        target_location = (
            target_wp.transform.location
        )

        dx = float(
            target_location.x
            - vehicle_location.x
        )
        dy = float(
            target_location.y
            - vehicle_location.y
        )

        distance_carla_m = math.sqrt(
            dx * dx + dy * dy
        )

        if distance_carla_m <= 1e-6:
            return current_error, 0.0

        target_bearing = math.atan2(
            dy,
            dx,
        )

        vehicle_yaw = math.radians(
            float(
                self.vehicle
                .get_transform()
                .rotation
                .yaw
            )
        )

        preview_error = self._wrap_pi(
            vehicle_yaw
            - target_bearing
        )

        return (
            float(preview_error),
            float(
                distance_carla_m
                / max(
                    float(
                        self.road_metrics.carla_scale
                    ),
                    1e-9,
                )
            ),
        )

    def oracle_steer_target_unit(self):
        """
        TRAIN-ONLY privileged pure-pursuit steering target.

        Uses the exact same forward Driving waypoint geometry as the v1.4
        reward lookahead.  This value is used only by reward shaping and
        diagnostics; it is never appended to PPO observation.
        """
        try:
            projected_wp = (
                self.road_metrics
                ._driving_waypoint_projected()
            )
            (
                preview_error_rad,
                distance_real_m,
            ) = self._preview_heading_error_rad(
                projected_wp
            )
        except Exception:
            return 0.0, 0.0, 0.0

        if float(distance_real_m) <= 1e-6:
            return (
                0.0,
                float(preview_error_rad),
                float(distance_real_m),
            )

        # _preview_heading_error_rad = vehicle_yaw - target_bearing.
        # Pure pursuit alpha uses target_bearing - vehicle_yaw.
        alpha_rad = -float(
            preview_error_rad
        )

        wheelbase_m = float(
            REAL.wheelbase_m
        )
        max_steer_rad = math.radians(
            float(
                REAL.max_steer_angle_deg
            )
        )

        delta_rad = math.atan2(
            2.0
            * wheelbase_m
            * math.sin(alpha_rad),
            max(
                float(distance_real_m),
                0.05,
            ),
        )

        steer_unit = (
            float(delta_rad)
            / max(
                float(max_steer_rad),
                1e-9,
            )
        )
        steer_unit = max(
            -1.0,
            min(
                1.0,
                float(steer_unit),
            ),
        )

        return (
            float(steer_unit),
            float(preview_error_rad),
            float(distance_real_m),
        )

    def reset(
        self,
        prev_steer_cmd=0.0,
        prev_speed_cmd_mps=0.0,
    ):
        self.elapsed_s = 0.0
        self.step_count = 0
        self.episode_reward = 0.0

        self.prev_steer_cmd = float(
            prev_steer_cmd
        )
        self.prev_speed_cmd_mps = float(
            prev_speed_cmd_mps
        )

        self.road_metrics.reset()
        self.collision_tracker.reset()

        # Initialize privileged previous errors from current pose so the
        # first reward step does not get a fake recovery-progress spike.
        try:
            projected_wp = (
                self.road_metrics
                ._driving_waypoint_projected()
            )
            self.prev_abs_lateral_error_m = abs(
                float(
                    self.road_metrics
                    ._lateral_error_real_m(
                        projected_wp
                    )
                )
            )
            (
                preview_heading_error_rad,
                _,
            ) = self._preview_heading_error_rad(
                projected_wp
            )
            self.prev_abs_heading_error_rad = abs(
                float(
                    preview_heading_error_rad
                )
            )
        except Exception:
            self.prev_abs_lateral_error_m = 0.0
            self.prev_abs_heading_error_rad = 0.0

        (
            state,
            _,
            _,
            _,
            _,
        ) = self.reward_fn.state_and_target(
            cruise_target_speed_mps=1.0,
            lateral_error_m=(
                self.prev_abs_lateral_error_m
            ),
            heading_error_rad=(
                self.prev_abs_heading_error_rad
            ),
        )

        self.prev_recovery_state = str(state)
        self.recovery_active = bool(
            state != RewardV5.STATE_STABLE
        )
        self.recovery_success_hold = 0
        self.recovery_event_count = (
            1 if self.recovery_active else 0
        )
        self.recovery_success_count = 0

    def _termination(
        self,
        collision,
        offroad,
        stuck,
        overspeed,
    ):
        if (
            self.episode_cfg.terminate_on_collision
            and collision
        ):
            return (
                True,
                False,
                "collision",
            )

        if (
            self.episode_cfg.terminate_on_offroad
            and offroad
        ):
            return (
                True,
                False,
                "offroad",
            )

        if (
            self.episode_cfg.terminate_on_stuck
            and stuck
        ):
            return (
                True,
                False,
                "stuck",
            )

        if overspeed:
            return (
                True,
                False,
                "overspeed",
            )

        if (
            self.elapsed_s
            >= self.episode_cfg.max_episode_seconds
        ):
            return (
                False,
                True,
                "time_limit",
            )

        return (
            False,
            False,
            None,
        )

    def _update_recovery_event_state(
        self,
        current_state,
    ):
        current_state = str(current_state)

        current_is_recovery = bool(
            current_state
            != RewardV5.STATE_STABLE
        )

        trigger_event = False
        success_event = False

        if (
            current_is_recovery
            and not self.recovery_active
        ):
            self.recovery_active = True
            self.recovery_success_hold = 0
            self.recovery_event_count += 1
            trigger_event = True

        if self.recovery_active:
            if (
                current_state
                == RewardV5.STATE_STABLE
            ):
                self.recovery_success_hold += 1
            else:
                self.recovery_success_hold = 0

            if (
                self.recovery_success_hold
                >= int(
                    self.reward_fn.cfg
                    .recovery_success_hold_ticks
                )
            ):
                success_event = True
                self.recovery_active = False
                self.recovery_success_hold = 0
                self.recovery_success_count += 1

        self.prev_recovery_state = current_state

        return (
            trigger_event,
            success_event,
        )

    def step(
        self,
        speed_mps,
        steer_cmd,
        speed_cmd_mps,
        dt,
        desired_speed_mps=None,
        teacher_steer_target_unit=0.0,
    ):
        """
        Call AFTER action application + world.tick().

        desired_speed_mps:
            cruise ceiling chosen by task/curriculum.
            RewardV5 derives state target from this cruise ceiling.

        Returns:
            reward, done, info
        """
        dt = max(float(dt), 0.0)

        speed_mps = float(speed_mps)
        steer_cmd = float(steer_cmd)
        speed_cmd_mps = float(
            speed_cmd_mps
        )

        if desired_speed_mps is None:
            desired_speed_mps = (
                speed_cmd_mps
            )

        desired_speed_mps = float(
            desired_speed_mps
        )

        self.elapsed_s += dt
        self.step_count += 1

        collision = (
            self.collision_tracker
            .consume_collision()
        )

        metrics = self.road_metrics.update(
            speed_mps=speed_mps,
            target_speed_cmd_mps=(
                speed_cmd_mps
            ),
            dt=dt,
            collision=collision,
        )

        abs_lat = abs(
            float(
                metrics[
                    "lateral_error_m"
                ]
            )
        )
        current_heading_error_rad = float(
            metrics[
                "heading_error_rad"
            ]
        )

        try:
            projected_wp = (
                self.road_metrics
                ._driving_waypoint_projected()
            )
            (
                preview_heading_error_rad,
                preview_distance_m,
            ) = self._preview_heading_error_rad(
                projected_wp
            )
        except Exception:
            preview_heading_error_rad = float(
                current_heading_error_rad
            )
            preview_distance_m = 0.0

        abs_heading = abs(
            float(
                preview_heading_error_rad
            )
        )

        (
            recovery_state,
            state_target_speed_mps,
            severity,
            lat_severity,
            heading_severity,
        ) = self.reward_fn.state_and_target(
            cruise_target_speed_mps=(
                desired_speed_mps
            ),
            lateral_error_m=abs_lat,
            heading_error_rad=abs_heading,
        )

        (
            recovery_trigger_event,
            recovery_success_event,
        ) = self._update_recovery_event_state(
            recovery_state
        )

        # Original repository also terminates a car that is effectively not
        # moving after 10 seconds. Scale its 1 km/h vs 22 km/h threshold to
        # the SIMPLE target speed instead of relying on PPO's commanded speed.
        idle_speed_threshold_mps = max(0.01, float(desired_speed_mps) / 22.0)
        idle_stuck = bool(
            self.elapsed_s >= 10.0
            and speed_mps < idle_speed_threshold_mps
        )
        effective_stuck = bool(metrics["stuck"] or idle_stuck)

        # Original-repository-style hard overspeed termination.
        _, _, max_speed_mps = self.reward_fn.speed_limits(
            desired_speed_mps
        )
        overspeed = bool(speed_mps > float(max_speed_mps))

        reward, terms = self.reward_fn.compute(
            progress_delta_m=(
                metrics[
                    "forward_progress_m"
                ]
            ),
            lateral_error_m=abs_lat,
            heading_error_rad=abs_heading,
            prev_lateral_error_m=(
                self.prev_abs_lateral_error_m
            ),
            prev_heading_error_rad=(
                self.prev_abs_heading_error_rad
            ),
            speed_mps=speed_mps,
            cruise_target_speed_mps=(
                desired_speed_mps
            ),
            steer_cmd=steer_cmd,
            prev_steer_cmd=(
                self.prev_steer_cmd
            ),
            speed_cmd_mps=speed_cmd_mps,
            prev_speed_cmd_mps=(
                self.prev_speed_cmd_mps
            ),
            dt=dt,
            teacher_steer_target_unit=float(
                teacher_steer_target_unit
            ),
            recovery_success_event=(
                recovery_success_event
            ),
            collision=metrics["collision"],
            offroad=metrics["offroad"],
            stuck=effective_stuck,
            overspeed=overspeed,
        )

        (
            terminated,
            truncated,
            reason,
        ) = self._termination(
            collision=metrics["collision"],
            offroad=metrics["offroad"],
            stuck=effective_stuck,
            overspeed=overspeed,
        )

        done = bool(
            terminated or truncated
        )

        self.prev_steer_cmd = float(
            steer_cmd
        )
        self.prev_speed_cmd_mps = float(
            speed_cmd_mps
        )
        self.prev_abs_lateral_error_m = float(
            abs_lat
        )
        self.prev_abs_heading_error_rad = float(
            abs_heading
        )

        self.episode_reward += float(
            reward
        )

        info = {
            # ----------------------------------------------------------
            # Reward
            # ----------------------------------------------------------
            "reward_terms": terms,
            "final_reward": float(reward),
            "episode_reward": float(
                self.episode_reward
            ),
            "teacher_steer_target_unit": float(
                terms.get(
                    "_teacher_steer_target",
                    0.0,
                )
            ),
            "teacher_steer_error": float(
                terms.get(
                    "_teacher_steer_error",
                    0.0,
                )
            ),
            "teacher_steer_factor": float(
                terms.get(
                    "_teacher_steer_factor",
                    0.0,
                )
            ),

            # ----------------------------------------------------------
            # State-aware speed
            # ----------------------------------------------------------
            "desired_speed_mps": float(
                desired_speed_mps
            ),
            "recovery_state": str(
                recovery_state
            ),
            "recovery_active": bool(
                self.recovery_active
            ),

            "adaptive_target_speed_mps": float(
                state_target_speed_mps
            ),
            "adaptive_speed_severity": float(
                severity
            ),
            "adaptive_lat_severity": float(
                lat_severity
            ),
            "adaptive_heading_severity": float(
                heading_severity
            ),

            "speed_mps": float(
                speed_mps
            ),
            "speed_cmd_mps": float(
                speed_cmd_mps
            ),
            "simple_min_speed_mps": float(
                self.reward_fn.speed_limits(desired_speed_mps)[0]
            ),
            "simple_target_speed_mps": float(
                self.reward_fn.speed_limits(desired_speed_mps)[1]
            ),
            "simple_max_speed_mps": float(
                max_speed_mps
            ),
            "overspeed": bool(overspeed),

            # ----------------------------------------------------------
            # Recovery event diagnostics
            # ----------------------------------------------------------
            "recovery_trigger_event": bool(
                recovery_trigger_event
            ),
            "recovery_success_event": bool(
                recovery_success_event
            ),
            "recovery_success_hold": int(
                self.recovery_success_hold
            ),
            "recovery_event_count": int(
                self.recovery_event_count
            ),
            "recovery_success_count": int(
                self.recovery_success_count
            ),

            # ----------------------------------------------------------
            # Road metrics
            # ----------------------------------------------------------
            "lateral_error_m": float(
                metrics[
                    "lateral_error_m"
                ]
            ),
            # Current lane-tangent heading remains diagnostic only.
            "heading_error_rad": float(
                current_heading_error_rad
            ),
            # Reward heading is anticipatory pure-pursuit style.
            "preview_heading_error_rad": float(
                preview_heading_error_rad
            ),
            "preview_lookahead_distance_m": float(
                preview_distance_m
            ),
            "configured_heading_lookahead_m": float(
                getattr(
                    self.reward_fn.cfg,
                    "heading_lookahead_m",
                    0.0,
                )
            ),
            "forward_progress_m": float(
                metrics[
                    "forward_progress_m"
                ]
            ),

            "collision": bool(
                metrics["collision"]
            ),
            "offroad": bool(
                metrics["offroad"]
            ),
            "center_offroad": bool(
                metrics[
                    "center_offroad"
                ]
            ),

            "stuck": bool(
                effective_stuck
            ),
            "controller_stuck": bool(
                metrics["stuck"]
            ),
            "idle_stuck": bool(idle_stuck),
            "idle_speed_threshold_mps": float(idle_speed_threshold_mps),
            "stuck_time_s": float(
                metrics[
                    "stuck_time_s"
                ]
            ),

            # Wheel-aware diagnostics.
            "wheel_out_count": int(
                metrics[
                    "wheel_out_count"
                ]
            ),
            "wheel_out_flags": tuple(
                metrics[
                    "wheel_out_flags"
                ]
            ),
            "wheel_offroad_consecutive_ticks": int(
                metrics[
                    "wheel_offroad_consecutive_ticks"
                ]
            ),
            "wheel_offroad_confirm_ticks": int(
                metrics[
                    "wheel_offroad_confirm_ticks"
                ]
            ),
            "wheel_offroad_min_points_out": int(
                metrics[
                    "wheel_offroad_min_points_out"
                ]
            ),
            "wheel_point_source": str(
                metrics[
                    "wheel_point_source"
                ]
            ),
            "offroad_mode": (
                "V5_2of4_points_3ticks"
            ),

            # ----------------------------------------------------------
            # Episode
            # ----------------------------------------------------------
            "episode_time_s": float(
                self.elapsed_s
            ),
            "episode_steps": int(
                self.step_count
            ),

            "terminated": bool(
                terminated
            ),
            "truncated": bool(
                truncated
            ),
            "done": bool(done),
            "termination_reason": reason,
        }

        return (
            float(reward),
            bool(done),
            info,
        )

    def destroy(self):
        if self.collision_tracker is not None:
            self.collision_tracker.destroy()
            self.collision_tracker = None
