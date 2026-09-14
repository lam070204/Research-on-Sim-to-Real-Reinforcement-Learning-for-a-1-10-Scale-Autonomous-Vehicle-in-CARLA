# -*- coding: utf-8 -*-
"""
CARLA RGB V5 SIMPLE / MULTI-MAP / MIXED DR

Environment V5 cho PPO multi-map: clean spawn, balanced spawn, delayed recovery.

Không inherit V3/V4.
Không monkey-patch module cũ.
Không cấu hình reward V3 từ environment.
Không cộng recovery reward delta ở environment.

Pipeline mỗi step:
    PPO action
      -> optional steering disturbance
      -> DomainRandomizedActionControllerV5
      -> CARLA tick
      -> sensor read
      -> obs100
      -> RewardManagerV5
      -> FINAL reward
      -> return

Observation:
    torch.float32, shape (100,)
    [latent95, speed, yaw_rate, accel_x, prev_steer, prev_speed_cmd]

Action:
    [steer_cmd, speed_cmd_mps]
    steer_cmd       in [-1, +1]
    speed_cmd_mps   in [0, REAL.max_speed_mps]

Privileged lateral/heading/wheel state chỉ dùng reward/termination.

Python 3.7 compatible.
"""

from __future__ import print_function

import copy
import math
import random
import time
import traceback

import numpy as np

from simulation.carla_connection_v5 import carla
from simulation.carla_sensors_v5 import (
    CameraSensor,
    IMUSensorV5,
    _configure_front_rgb_blueprint,
)

from action_controller_v5 import ActionControllerV5

from domain_randomization_v5_simple import (
    nominal_domain_parameters_v5,
    sample_domain_randomization_v5,
    apply_domain_physics_v5,
    DomainRandomizedActionControllerV5,
    nominal_vision_domain_v5,
    sample_vision_domain_v5,
    augment_camera_rgb_v5,
    noisy_state_for_policy_v5,
)

from encoder_runtime_rgb_v5 import EncodeRGBV5
from observation_builder_rgb_v5 import (
    ObservationBuilderRGBV5,
    OBSERVATION_DIM_V5,
)

from reward_manager_v5_simple import (
    RewardManagerV5,
    EpisodeConfigV5,
)

from vehicle_specs_v5 import (
    VEHICLE_BLUEPRINT_ID,
    REAL,
    CARLA_GEOMETRY_SCALE,
    CARLA_WHEEL_RADIUS_CM,
    CARLA_FRONT_MAX_STEER_DEG,
    CARLA_REAR_MAX_STEER_DEG,
    FRONT_CAMERA_X,
    FRONT_CAMERA_Y,
    FRONT_CAMERA_Z,
    FRONT_CAMERA_PITCH_DEG,
    FRONT_CAMERA_YAW_DEG,
    FRONT_CAMERA_ROLL_DEG,
    FRONT_CAMERA_FOV_DEG,
    FRONT_CAMERA_FPS,
)


CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Reward V5 state targets đang được thiết kế theo cruise ceiling = 1.0 m/s.
DEFAULT_DESIRED_SPEED_MPS = 1.00
DEFAULT_MAX_EPISODE_SECONDS = 60.0

DEFAULT_SPAWN_Z_OFFSET_M = 0.25
DEFAULT_SETTLE_SECONDS = 3.0

SENSOR_CALLBACK_TIMEOUT_S = 3.0
CAMERA_MAX_AGE_S = 0.080
CAMERA_MAX_AGE_TICKS = int(
    math.ceil(CAMERA_MAX_AGE_S / CONTROL_DT)
)
SENSOR_PRIME_MAX_TICKS = 50

DESTROY_DRAIN_TICKS = 2
ACTOR_DEBUG_EVERY_RESETS = 25

CAMERA_LAG_LOG_FIRST_N = 10
CAMERA_LAG_LOG_EVERY = 100


def _clip(value, low, high):
    return float(
        max(
            float(low),
            min(float(high), float(value)),
        )
    )


def _to_action_array(action):
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()

    array = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    if array.shape != (2,):
        raise ValueError(
            "Action V5 phải có shape (2,), nhận được {}.".format(
                tuple(array.shape)
            )
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Action V5 chứa NaN/Inf: {}".format(
                array
            )
        )

    return array


class RecoveryScenarioConfigV5(object):
    """
    Chỉ quản lý PHÂN PHỐI training state.

    Reward logic KHÔNG nằm ở đây.
    Reward state/target nằm trong reward_function_v5.py.
    """

    def __init__(self):
        # MULTI-SPAWN curriculum: spawn perturbation is permanently disabled.
        # Recovery is injected only while driving, after a stability gate.
        self.recovery_spawn_probability = 0.0

        # Kept only for backward-compatible config parsing. They are not
        # applied to the spawn transform in ONE-SPAWN mode.
        self.spawn_lateral_min_m = 0.0
        self.spawn_lateral_max_m = 0.0
        self.spawn_heading_min_deg = 0.0
        self.spawn_heading_max_deg = 0.0

        # Mid-episode steering disturbance curriculum.
        self.disturbance_probability = 1.0
        # Deprecated legacy fields retained so old trainer args do not break.
        self.disturbance_start_min_s = 0.0
        self.disturbance_start_max_s = 0.0
        self.disturbance_duration_min_s = 0.10
        self.disturbance_duration_max_s = 0.16
        self.disturbance_steer_min = 0.10
        self.disturbance_steer_max = 0.18

        # Do not inject recovery until the car has learned to drive for a while.
        # At 50 Hz: 700 ticks ~= 14 seconds.
        self.recovery_min_episode_step = 700
        self.recovery_stable_ticks = 100
        self.recovery_stable_lateral_m = 0.040
        self.recovery_stable_heading_deg = 5.0
        self.recovery_stable_speed_mps = 0.40
        self.recovery_cooldown_ticks = 400
        self.recovery_max_events_per_episode = 1
        self.recovery_success_hold_ticks = 8

        self.seed = 505


RECOVERY_CONFIG_V5 = RecoveryScenarioConfigV5()


def configure_recovery_scenarios_v5(**kwargs):
    for key, value in kwargs.items():
        if not hasattr(RECOVERY_CONFIG_V5, key):
            raise ValueError(
                "Unknown V5 recovery config field: {}".format(
                    key
                )
            )

        setattr(
            RECOVERY_CONFIG_V5,
            key,
            value,
        )


class CameraSensorVisionDRV5(CameraSensor):
    """
    CameraSensor V5 với transform/FOV được sample theo episode.
    """

    def __init__(
        self,
        vehicle,
        camera_transform,
        fov_deg,
    ):
        self._v5_camera_transform = camera_transform
        self._v5_fov_deg = float(fov_deg)

        super(
            CameraSensorVisionDRV5,
            self,
        ).__init__(vehicle)

    def _set_camera_sensor(self, world):
        camera_bp = (
            world.get_blueprint_library()
            .find(self.sensor_name)
        )

        _configure_front_rgb_blueprint(
            camera_bp
        )

        camera_bp.set_attribute(
            "fov",
            str(float(self._v5_fov_deg)),
        )

        camera_bp.set_attribute(
            "sensor_tick",
            str(
                1.0
                / float(FRONT_CAMERA_FPS)
            ),
        )

        return world.spawn_actor(
            camera_bp,
            self._v5_camera_transform,
            attach_to=self.parent,
            attachment_type=(
                carla.AttachmentType.Rigid
            ),
        )


class CarlaEnvironmentRGBV5(object):
    """
    Canonical clean V5 environment.

    reset() ->
        observation tensor (100,)

    step(action[2]) ->
        observation tensor (100,),
        reward float,
        done bool,
        info dict
    """

    def __init__(
        self,
        client,
        world,
        town=None,
        safe_spawn_numbers=None,
        control_hz=CONTROL_HZ,
        desired_speed_mps=DEFAULT_DESIRED_SPEED_MPS,
        max_episode_seconds=DEFAULT_MAX_EPISODE_SECONDS,
        encoder_device=None,
        spawn_z_offset_m=DEFAULT_SPAWN_Z_OFFSET_M,
        settle_seconds=DEFAULT_SETTLE_SECONDS,
        steer_rate_limit_per_s=None,
        speed_rate_limit_mps2=None,

        # Clean explicit DR switches.
        dynamics_dr_enabled=True,
        vision_dr_enabled=True,
        sensor_dr_enabled=True,
        domain_randomization_seed=None,
        # All DR families are available from step 0, but only this fraction
        # of episodes use the randomized domain. The rest are exact nominal
        # episodes so PPO cannot learn a permanent compensation for noise.
        dr_episode_probability=0.75,

        # SIMPLE baseline: recovery injection is disabled by default.
        recovery_scenarios_enabled=False,
    ):
        self.client = client
        self.world = world
        self.town = town

        self.map = self.world.get_map()
        self.blueprint_library = (
            self.world.get_blueprint_library()
        )

        self.control_hz = float(control_hz)

        if self.control_hz <= 0.0:
            raise ValueError(
                "control_hz phải > 0."
            )

        if abs(
            self.control_hz - CONTROL_HZ
        ) > 1e-6:
            raise ValueError(
                "V5 khóa control_hz = 50 Hz."
            )

        self.dt = (
            1.0 / self.control_hz
        )

        self.desired_speed_mps = (
            self._sanitize_desired_speed(
                desired_speed_mps
            )
        )

        self.max_episode_seconds = float(
            max_episode_seconds
        )

        self.spawn_z_offset_m = float(
            spawn_z_offset_m
        )
        self.settle_seconds = float(
            settle_seconds
        )

        self.steer_rate_limit_per_s = (
            steer_rate_limit_per_s
        )
        self.speed_rate_limit_mps2 = (
            speed_rate_limit_mps2
        )

        # --------------------------------------------------------------
        # DR
        # --------------------------------------------------------------
        self.dynamics_dr_enabled = bool(
            dynamics_dr_enabled
        )
        self.vision_dr_enabled = bool(
            vision_dr_enabled
        )
        self.sensor_dr_enabled = bool(
            sensor_dr_enabled
        )

        self.domain_randomization_seed = (
            domain_randomization_seed
        )

        self.dr_episode_probability = _clip(
            dr_episode_probability,
            0.0,
            1.0,
        )
        self._episode_dr_active = False

        self._domain_rng = random.Random(
            domain_randomization_seed
        )

        if domain_randomization_seed is None:
            self._vision_rng = random.Random()
        else:
            self._vision_rng = random.Random(
                int(domain_randomization_seed)
                + 44004
            )

        self.domain_randomization_params = (
            nominal_domain_parameters_v5()
        )
        self.domain_randomization_applied = None

        self.vision_sensor_params = (
            nominal_vision_domain_v5()
        )
        self._vision_np_rng = (
            np.random.RandomState(0)
        )

        self.last_augmented_rgb = None
        self.last_policy_sensor_state = None
        self._last_augmented_camera_frame = None

        # --------------------------------------------------------------
        # Recovery scenarios
        # --------------------------------------------------------------
        self.recovery_scenarios_enabled = bool(
            recovery_scenarios_enabled
        )

        self._recovery_rng = random.Random(
            int(RECOVERY_CONFIG_V5.seed)
        )

        # Balanced disturbance sign scheduler. Each pair contains exactly
        # one negative and one positive disturbance, shuffled per pair.
        self._disturbance_sign_order = [-1.0, +1.0]
        self._recovery_rng.shuffle(self._disturbance_sign_order)
        self._disturbance_sign_cursor = 0

        self._episode_index = 0

        self._spawn_recovery = False
        self._spawn_offset_m = 0.0
        self._spawn_heading_deg = 0.0

        self._disturbance_enabled = False
        self._disturbance_start_tick = 0
        self._disturbance_end_tick = 0
        self._disturbance_steer = 0.0

        # Recovery curriculum runtime state.
        self._episode_step = 0
        self._stable_gate_ticks = 0
        self._last_disturbance_end_tick = -10 ** 9
        self._disturbance_event_count = 0
        self._disturbance_trigger_event = False
        self._disturbance_success_event = False
        self._disturbance_recovery_pending = False
        self._disturbance_recovery_success_hold = 0
        self._episode_selected_for_disturbance = False

        # --------------------------------------------------------------
        # World / spawn
        # --------------------------------------------------------------
        self._original_world_settings = (
            self.world.get_settings()
        )

        try:
            self._initial_weather = copy.copy(
                self.world.get_weather()
            )
        except Exception:
            self._initial_weather = None

        self._configure_synchronous_world()

        self.spawn_points = list(
            self.map.get_spawn_points()
        )

        if not self.spawn_points:
            raise RuntimeError(
                "Map hiện tại không có spawn point."
            )

        self.safe_spawn_indices = (
            self._resolve_spawn_indices(
                safe_spawn_numbers
            )
        )

        # CLEAN multi-spawn: location/heading are never perturbed at reset.
        # Spawn order is shuffled in balanced cycles: every valid spawn is used
        # exactly once before the next reshuffle.
        self.spawn_order = list(self.safe_spawn_indices)
        self._recovery_rng.shuffle(self.spawn_order)
        self.spawn_cursor = 0
        self.current_spawn_index = None
        self._forced_spawn_index = None

        # --------------------------------------------------------------
        # Frozen encoder / observation
        # --------------------------------------------------------------
        encoder = EncodeRGBV5(
            device=encoder_device
        )

        self.observation_builder = (
            ObservationBuilderRGBV5(
                encoder=encoder
            )
        )

        # --------------------------------------------------------------
        # Episode-owned actors/components
        # --------------------------------------------------------------
        self.vehicle = None
        self.camera_obj = None
        self.imu_obj = None

        self.action_controller = None
        self.reward_manager = None

        self.episode_done = True

        self.last_sim_frame = -1
        self.last_camera_frame = -1
        self.last_camera_frame_lag = None

        self.camera_lag_event_count = 0
        self.camera_max_observed_lag = 0

        self.reset_count = 0

        print(
            "ENV RGB V5 CLEAN | obs={} | action=[steer,speed] | "
            "Hz={:.1f} | cruise={:.3f} m/s | "
            "dynamicsDR={} visionDR={} sensorDR={} | "
            "DRmix={:.0f}% randomized / {:.0f}% nominal | "
            "recovery={} | spawns={}".format(
                OBSERVATION_DIM_V5,
                self.control_hz,
                self.desired_speed_mps,
                self.dynamics_dr_enabled,
                self.vision_dr_enabled,
                self.sensor_dr_enabled,
                100.0 * self.dr_episode_probability,
                100.0 * (1.0 - self.dr_episode_probability),
                self.recovery_scenarios_enabled,
                [
                    i + 1
                    for i
                    in self.safe_spawn_indices
                ],
            )
        )

    # ==================================================================
    # Basic config
    # ==================================================================

    def _sanitize_desired_speed(
        self,
        value,
    ):
        value = float(value)

        max_speed = float(
            REAL.max_speed_mps
        )

        if not np.isfinite(value):
            raise ValueError(
                "desired_speed_mps phải finite."
            )

        return float(
            np.clip(
                value,
                0.0,
                max_speed,
            )
        )

    def set_desired_speed_mps(
        self,
        value,
    ):
        self.desired_speed_mps = (
            self._sanitize_desired_speed(
                value
            )
        )

    def _configure_synchronous_world(self):
        settings = (
            self.world.get_settings()
        )

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            self.dt
        )

        self.world.apply_settings(
            settings
        )

        self.last_sim_frame = int(
            self.world.tick()
        )

    # ==================================================================
    # Spawn / recovery scenario sampling
    # ==================================================================

    def _resolve_spawn_indices(
        self,
        safe_spawn_numbers,
    ):
        if safe_spawn_numbers is None:
            # Default: first CARLA spawn only when caller does not provide a set.
            return [0]

        indices = []

        for spawn_number in (
            safe_spawn_numbers
        ):
            index = int(
                spawn_number
            ) - 1

            if (
                0
                <= index
                < len(self.spawn_points)
            ):
                indices.append(index)
            else:
                print(
                    "WARNING: bỏ spawn {} vì map chỉ có {} spawn.".format(
                        spawn_number,
                        len(self.spawn_points),
                    )
                )

        indices = list(
            dict.fromkeys(indices)
        )

        if not indices:
            raise RuntimeError(
                "safe_spawn_numbers không có spawn hợp lệ."
            )

        return indices

    def _sample_signed(
        self,
        low,
        high,
    ):
        """Balanced random sign: one + and one - per shuffled pair."""
        magnitude = self._recovery_rng.uniform(
            float(low),
            float(high),
        )

        if self._disturbance_sign_cursor >= len(self._disturbance_sign_order):
            self._disturbance_sign_order = [-1.0, +1.0]
            self._recovery_rng.shuffle(self._disturbance_sign_order)
            self._disturbance_sign_cursor = 0

        sign = float(
            self._disturbance_sign_order[self._disturbance_sign_cursor]
        )
        self._disturbance_sign_cursor += 1

        return float(sign * magnitude)

    def _sample_recovery_scenario(self):
        """
        CLEAN multi-spawn curriculum.

        Spawn pose is ALWAYS nominal: offset=0, heading=0.
        Recovery is not scheduled by absolute time here. Instead the step()
        stability gate decides when a mid-episode disturbance may start.
        """
        self._episode_index += 1
        self._episode_step = 0
        self._stable_gate_ticks = 0
        self._last_disturbance_end_tick = -10 ** 9
        self._disturbance_event_count = 0
        self._disturbance_trigger_event = False
        self._disturbance_success_event = False
        self._disturbance_recovery_pending = False
        self._disturbance_recovery_success_hold = 0

        # Spawn recovery is permanently disabled for sim-to-real nominal path.
        self._spawn_recovery = False
        self._spawn_offset_m = 0.0
        self._spawn_heading_deg = 0.0

        self._disturbance_enabled = False
        self._disturbance_start_tick = 0
        self._disturbance_end_tick = 0
        self._disturbance_steer = 0.0

        if not self.recovery_scenarios_enabled:
            self._episode_selected_for_disturbance = False
            return

        cfg = RECOVERY_CONFIG_V5
        self._episode_selected_for_disturbance = bool(
            self._recovery_rng.random()
            < float(cfg.disturbance_probability)
        )

    def _recovery_gate_state(self, clean_imu_state):
        """Return whether the car is inside the configured safe corridor."""
        cfg = RECOVERY_CONFIG_V5

        try:
            projected_wp = (
                self.reward_manager.road_metrics
                ._driving_waypoint_projected()
            )
            abs_lat = abs(float(
                self.reward_manager.road_metrics
                ._lateral_error_real_m(projected_wp)
            ))
            abs_heading = abs(float(
                self.reward_manager.road_metrics
                ._heading_error_rad(projected_wp)
            ))
        except Exception:
            return False, 0.0, 0.0, 0.0

        speed_mps = max(0.0, float(
            clean_imu_state.get("speed_mps", 0.0)
        ))

        stable_now = bool(
            abs_lat <= float(cfg.recovery_stable_lateral_m)
            and abs_heading <= math.radians(
                float(cfg.recovery_stable_heading_deg)
            )
            and speed_mps >= float(cfg.recovery_stable_speed_mps)
        )

        return stable_now, abs_lat, abs_heading, speed_mps

    def _update_recovery_curriculum_gate(self, clean_imu_state):
        """
        Delayed mid-episode recovery curriculum.

        1) Spawn is always clean.
        2) No disturbance before recovery_min_episode_step.
        3) Require a consecutive stable window.
        4) Inject a short steering disturbance.
        5) Mark success only after the disturbance has actually ended and the
           car returns to the safe corridor for several consecutive ticks.
        """
        self._disturbance_trigger_event = False
        self._disturbance_success_event = False

        if not self.recovery_scenarios_enabled:
            self._stable_gate_ticks = 0
            return

        cfg = RECOVERY_CONFIG_V5
        stable_now, _, _, _ = self._recovery_gate_state(
            clean_imu_state
        )

        # Finish an active injected disturbance first.
        if self._disturbance_enabled:
            if self._episode_step >= self._disturbance_end_tick:
                self._disturbance_enabled = False
                self._last_disturbance_end_tick = int(self._episode_step)
                self._disturbance_steer = 0.0
                self._disturbance_recovery_pending = True
                self._disturbance_recovery_success_hold = 0
            return

        # If a disturbance has completed, measure actual recovery success.
        if self._disturbance_recovery_pending:
            if stable_now:
                self._disturbance_recovery_success_hold += 1
            else:
                self._disturbance_recovery_success_hold = 0

            if self._disturbance_recovery_success_hold >= int(
                cfg.recovery_success_hold_ticks
            ):
                self._disturbance_success_event = True
                self._disturbance_recovery_pending = False
                self._disturbance_recovery_success_hold = 0
                print(
                    "RECOVERY SUCCESS | episode={} | step={} | event={}".format(
                        self._episode_index,
                        self._episode_step,
                        self._disturbance_event_count,
                    )
                )
            return

        # No recovery before the nominal-learning window.
        if self._episode_step < int(cfg.recovery_min_episode_step):
            self._stable_gate_ticks = 0
            return

        if not self._episode_selected_for_disturbance:
            self._stable_gate_ticks = 0
            return

        if self._disturbance_event_count >= int(
            cfg.recovery_max_events_per_episode
        ):
            self._stable_gate_ticks = 0
            return

        if (
            self._episode_step - self._last_disturbance_end_tick
            < int(cfg.recovery_cooldown_ticks)
        ):
            self._stable_gate_ticks = 0
            return

        if stable_now:
            self._stable_gate_ticks += 1
        else:
            self._stable_gate_ticks = 0

        if self._stable_gate_ticks < int(cfg.recovery_stable_ticks):
            return

        duration_s = self._recovery_rng.uniform(
            float(cfg.disturbance_duration_min_s),
            float(cfg.disturbance_duration_max_s),
        )
        duration_ticks = max(
            1,
            int(round(duration_s * self.control_hz)),
        )

        self._disturbance_start_tick = int(self._episode_step) + 1
        self._disturbance_end_tick = (
            self._disturbance_start_tick + duration_ticks
        )
        self._disturbance_steer = self._sample_signed(
            cfg.disturbance_steer_min,
            cfg.disturbance_steer_max,
        )
        self._disturbance_enabled = True
        self._disturbance_event_count += 1
        self._disturbance_trigger_event = True
        self._stable_gate_ticks = 0

        print(
            "RECOVERY INJECT | episode={} | trigger_step={} | apply_step={} | "
            "duration={}ticks | steer={:+.3f} | event={}/{}".format(
                self._episode_index,
                self._episode_step,
                self._disturbance_start_tick,
                duration_ticks,
                self._disturbance_steer,
                self._disturbance_event_count,
                int(cfg.recovery_max_events_per_episode),
            )
        )

    def force_next_spawn_number(self, spawn_number):
        """Force exactly one upcoming reset to use a chosen 1-based spawn."""
        index = int(spawn_number) - 1
        if index not in self.safe_spawn_indices:
            raise ValueError(
                "Forced spawn {} is not in safe spawns {}.".format(
                    spawn_number,
                    [i + 1 for i in self.safe_spawn_indices],
                )
            )
        self._forced_spawn_index = int(index)

    def _next_balanced_spawn_index(self):
        if self._forced_spawn_index is not None:
            index = int(self._forced_spawn_index)
            self._forced_spawn_index = None
            return index

        if self.spawn_cursor >= len(self.spawn_order):
            self.spawn_order = list(self.safe_spawn_indices)
            self._recovery_rng.shuffle(self.spawn_order)
            self.spawn_cursor = 0

        index = int(self.spawn_order[self.spawn_cursor])
        self.spawn_cursor += 1
        return index

    def _next_base_spawn_transform(self):
        index = self._next_balanced_spawn_index()
        self.current_spawn_index = int(index)

        original = self.spawn_points[index]

        return carla.Transform(
            carla.Location(
                x=original.location.x,
                y=original.location.y,
                z=(
                    original.location.z
                    + self.spawn_z_offset_m
                ),
            ),
            original.rotation,
        )

    def _next_spawn_transform(self):
        # Clean-spawn guarantee: recovery never changes spawn position/heading.
        return self._next_base_spawn_transform()

    # ==================================================================
    # Vehicle
    # ==================================================================

    def _get_vehicle_blueprint(self):
        matches = (
            self.blueprint_library
            .filter(
                VEHICLE_BLUEPRINT_ID
            )
        )

        if not matches:
            raise RuntimeError(
                "Không tìm thấy blueprint {}.".format(
                    VEHICLE_BLUEPRINT_ID
                )
            )

        return matches[0]

    def _spawn_vehicle(self):
        vehicle_bp = (
            self._get_vehicle_blueprint()
        )

        for _ in range(
            len(self.safe_spawn_indices)
        ):
            transform = (
                self._next_spawn_transform()
            )

            vehicle = (
                self.world.try_spawn_actor(
                    vehicle_bp,
                    transform,
                )
            )

            if vehicle is not None:
                return vehicle

            print(
                "Spawn {} occupied/invalid.".format(
                    self.current_spawn_index + 1
                )
            )

        raise RuntimeError(
            "Không spawn được {} tại các spawn đã chọn.".format(
                VEHICLE_BLUEPRINT_ID
            )
        )

    def _apply_vehicle_geometry(self):
        if self.vehicle is None:
            raise RuntimeError(
                "Vehicle chưa được spawn."
            )

        if (
            self.vehicle.type_id
            != VEHICLE_BLUEPRINT_ID
        ):
            raise RuntimeError(
                "Sai blueprint: {} != {}".format(
                    self.vehicle.type_id,
                    VEHICLE_BLUEPRINT_ID,
                )
            )

        physics = (
            self.vehicle
            .get_physics_control()
        )

        wheels = list(
            physics.wheels
        )

        if len(wheels) < 4:
            raise RuntimeError(
                "Vehicle trả về {} wheels, cần 4.".format(
                    len(wheels)
                )
            )

        for wheel in wheels[:4]:
            wheel.radius = float(
                CARLA_WHEEL_RADIUS_CM
            )

        wheels[0].max_steer_angle = float(
            CARLA_FRONT_MAX_STEER_DEG
        )
        wheels[1].max_steer_angle = float(
            CARLA_FRONT_MAX_STEER_DEG
        )
        wheels[2].max_steer_angle = float(
            CARLA_REAR_MAX_STEER_DEG
        )
        wheels[3].max_steer_angle = float(
            CARLA_REAR_MAX_STEER_DEG
        )

        physics.wheels = wheels

        self.vehicle.apply_physics_control(
            physics
        )

        self.last_sim_frame = int(
            self.world.tick()
        )

        applied = list(
            self.vehicle
            .get_physics_control()
            .wheels
        )

        radius_ok = all(
            abs(
                float(applied[i].radius)
                - float(
                    CARLA_WHEEL_RADIUS_CM
                )
            )
            < 1e-3
            for i in range(4)
        )

        steer_ok = (
            abs(
                float(
                    applied[0]
                    .max_steer_angle
                )
                - float(
                    CARLA_FRONT_MAX_STEER_DEG
                )
            )
            < 1e-3
            and
            abs(
                float(
                    applied[1]
                    .max_steer_angle
                )
                - float(
                    CARLA_FRONT_MAX_STEER_DEG
                )
            )
            < 1e-3
            and
            abs(
                float(
                    applied[2]
                    .max_steer_angle
                )
                - float(
                    CARLA_REAR_MAX_STEER_DEG
                )
            )
            < 1e-3
            and
            abs(
                float(
                    applied[3]
                    .max_steer_angle
                )
                - float(
                    CARLA_REAR_MAX_STEER_DEG
                )
            )
            < 1e-3
        )

        if not radius_ok or not steer_ok:
            raise RuntimeError(
                "Wheel radius/steering không giữ đúng sau apply."
            )

    # ==================================================================
    # DR
    # ==================================================================

    def _prepare_episode_dynamics_dr(self):
        if self.dynamics_dr_enabled and self._episode_dr_active:
            params = (
                sample_domain_randomization_v5(
                    rng=self._domain_rng
                )
            )
        else:
            params = (
                nominal_domain_parameters_v5()
            )

        self.domain_randomization_params = (
            dict(params)
        )

        self.domain_randomization_applied = (
            apply_domain_physics_v5(
                vehicle=self.vehicle,
                params=(
                    self.domain_randomization_params
                ),
                carla=carla,
            )
        )

        self.last_sim_frame = int(
            self.world.tick()
        )

        verify = (
            self.vehicle
            .get_physics_control()
        )

        expected_brake = float(
            self.domain_randomization_params[
                "max_brake_torque"
            ]
        )

        brake_ok = all(
            abs(
                float(
                    w.max_brake_torque
                )
                - expected_brake
            )
            < 1e-3
            for w in list(verify.wheels)
        )

        autobox_ok = (
            bool(
                verify.use_gear_autobox
            )
            == bool(
                self.domain_randomization_params[
                    "use_gear_autobox"
                ]
            )
        )

        gear_switch_ok = (
            abs(
                float(
                    verify.gear_switch_time
                )
                - float(
                    self.domain_randomization_params[
                        "gear_switch_time_s"
                    ]
                )
            )
            < 1e-6
        )

        if (
            not brake_ok
            or not autobox_ok
            or not gear_switch_ok
        ):
            raise RuntimeError(
                "V5 PhysicsControl readback mismatch."
            )

    def _sample_vision_sensor_episode(self):
        self.last_augmented_rgb = None
        self.last_policy_sensor_state = None
        self._last_augmented_camera_frame = None

        if (
            not self._episode_dr_active
            or (
                not self.vision_dr_enabled
                and not self.sensor_dr_enabled
            )
        ):
            self.vision_sensor_params = (
                nominal_vision_domain_v5()
            )
            self._vision_np_rng = (
                np.random.RandomState(0)
            )
            return

        sampled = sample_vision_domain_v5(
            rng=self._vision_rng
        )

        nominal = (
            nominal_vision_domain_v5()
        )

        # Disable only camera/weather/image parts if requested.
        if not self.vision_dr_enabled:
            vision_keys = (
                "camera_dx_real_m",
                "camera_dy_real_m",
                "camera_dz_real_m",
                "camera_dx_carla_m",
                "camera_dy_carla_m",
                "camera_dz_carla_m",
                "camera_pitch_offset_deg",
                "camera_yaw_offset_deg",
                "camera_roll_offset_deg",
                "camera_fov_offset_deg",
                "brightness_gain",
                "contrast_gain",
                "gamma",
                "red_gain",
                "green_gain",
                "blue_gain",
                "gaussian_noise_sigma",
            )

            for key in vision_keys:
                sampled[key] = nominal[key]

        # Disable only proprioceptive noise if requested.
        if not self.sensor_dr_enabled:
            sensor_keys = (
                "speed_bias_mps",
                "speed_noise_std_mps",
                "yaw_bias_rad_s",
                "yaw_noise_std_rad_s",
                "ax_bias_mps2",
                "ax_noise_std_mps2",
            )

            for key in sensor_keys:
                sampled[key] = nominal[key]

        self.vision_sensor_params = (
            dict(sampled)
        )

        self._vision_np_rng = (
            np.random.RandomState(
                int(
                    sampled[
                        "episode_noise_seed"
                    ]
                )
            )
        )

    def _make_camera_transform(self):
        p = self.vision_sensor_params

        return carla.Transform(
            carla.Location(
                x=(
                    float(FRONT_CAMERA_X)
                    + float(
                        p[
                            "camera_dx_carla_m"
                        ]
                    )
                ),
                y=(
                    float(FRONT_CAMERA_Y)
                    + float(
                        p[
                            "camera_dy_carla_m"
                        ]
                    )
                ),
                z=(
                    float(FRONT_CAMERA_Z)
                    + float(
                        p[
                            "camera_dz_carla_m"
                        ]
                    )
                ),
            ),
            carla.Rotation(
                pitch=(
                    float(
                        FRONT_CAMERA_PITCH_DEG
                    )
                    + float(
                        p[
                            "camera_pitch_offset_deg"
                        ]
                    )
                ),
                yaw=(
                    float(
                        FRONT_CAMERA_YAW_DEG
                    )
                    + float(
                        p[
                            "camera_yaw_offset_deg"
                        ]
                    )
                ),
                roll=(
                    float(
                        FRONT_CAMERA_ROLL_DEG
                    )
                    + float(
                        p[
                            "camera_roll_offset_deg"
                        ]
                    )
                ),
            ),
        )

    def _apply_weather(self):
        # Exact nominal episode: restore the world weather captured when the
        # environment was created. This prevents a previous randomized
        # episode from leaking weather into a nominal episode.
        if not (self.vision_dr_enabled and self._episode_dr_active):
            try:
                if self._initial_weather is not None:
                    self.world.set_weather(copy.copy(self._initial_weather))
            except Exception:
                pass
            return

        p = self.vision_sensor_params

        try:
            weather = (
                self.world.get_weather()
            )

            for key in (
                "cloudiness",
                "precipitation",
                "wetness",
                "fog_density",
                "wind_intensity",
                "sun_altitude_angle",
                "sun_azimuth_angle",
            ):
                if hasattr(weather, key):
                    setattr(
                        weather,
                        key,
                        float(p[key]),
                    )

            self.world.set_weather(
                weather
            )

        except Exception as error:
            print(
                "V5 WARNING | weather DR not applied: {}".format(
                    error
                )
            )

    # ==================================================================
    # Episode components
    # ==================================================================

    def _settle_vehicle(self):
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
            )
        )

        self.vehicle.set_target_velocity(
            carla.Vector3D(
                0.0,
                0.0,
                0.0,
            )
        )

        self.vehicle.set_target_angular_velocity(
            carla.Vector3D(
                0.0,
                0.0,
                0.0,
            )
        )

        n_ticks = max(
            1,
            int(
                round(
                    self.settle_seconds
                    * self.control_hz
                )
            ),
        )

        for _ in range(n_ticks):
            self.last_sim_frame = int(
                self.world.tick()
            )

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
            )
        )

    def _spawn_episode_components(self):
        if self.vision_dr_enabled and self._episode_dr_active:
            camera_transform = (
                self._make_camera_transform()
            )

            fov_deg = (
                float(FRONT_CAMERA_FOV_DEG)
                + float(
                    self.vision_sensor_params[
                        "camera_fov_offset_deg"
                    ]
                )
            )

            self.camera_obj = (
                CameraSensorVisionDRV5(
                    vehicle=self.vehicle,
                    camera_transform=(
                        camera_transform
                    ),
                    fov_deg=fov_deg,
                )
            )
        else:
            self.camera_obj = (
                CameraSensor(
                    self.vehicle
                )
            )

        self.imu_obj = (
            IMUSensorV5(
                self.vehicle
            )
        )

        base_action_controller = (
            ActionControllerV5(
                vehicle=self.vehicle,
                steer_rate_limit_per_s=(
                    self.steer_rate_limit_per_s
                ),
                speed_rate_limit_mps2=(
                    self.speed_rate_limit_mps2
                ),
            )
        )

        self.action_controller = (
            DomainRandomizedActionControllerV5(
                base_controller=(
                    base_action_controller
                ),
                steer_gain_positive=float(
                    self.domain_randomization_params[
                        "steer_gain_positive"
                    ]
                ),
                steer_gain_negative=float(
                    self.domain_randomization_params[
                        "steer_gain_negative"
                    ]
                ),
                extra_command_delay_ticks=int(
                    self.domain_randomization_params[
                        "extra_command_delay_ticks"
                    ]
                ),
            )
        )

        self.action_controller.reset()

        self.reward_manager = (
            RewardManagerV5(
                vehicle=self.vehicle,
                episode_config=(
                    EpisodeConfigV5(
                        max_episode_seconds=(
                            self.max_episode_seconds
                        ),
                        terminate_on_collision=True,
                        terminate_on_offroad=True,
                        terminate_on_stuck=True,
                    )
                ),
            )
        )

    # ==================================================================
    # Sensor timing
    # ==================================================================

    def _drain_destroyed_actors(self):
        for _ in range(
            DESTROY_DRAIN_TICKS
        ):
            self.last_sim_frame = int(
                self.world.tick()
            )

    def _debug_actor_counts(
        self,
        force=False,
    ):
        if (
            not force
            and self.reset_count
            % ACTOR_DEBUG_EVERY_RESETS
            != 0
        ):
            return

        try:
            actors = (
                self.world.get_actors()
            )

            vehicles = list(
                actors.filter(
                    "vehicle.*"
                )
            )

            sensors = list(
                actors.filter(
                    "sensor.*"
                )
            )

            sensor_types = {}

            for actor in sensors:
                sensor_types[
                    actor.type_id
                ] = (
                    sensor_types.get(
                        actor.type_id,
                        0,
                    )
                    + 1
                )

            print(
                "ACTOR DEBUG | reset={} | vehicles={} | sensors={} | "
                "sensor_types={}".format(
                    self.reset_count,
                    len(vehicles),
                    len(sensors),
                    sensor_types,
                )
            )

        except Exception as error:
            print(
                "ACTOR DEBUG FAILED | reset={} | error={}".format(
                    self.reset_count,
                    repr(error),
                )
            )

    def _log_camera_lag(
        self,
        sim_frame,
        camera_frame,
        lag,
    ):
        if lag <= 0:
            return

        self.camera_lag_event_count += 1

        self.camera_max_observed_lag = max(
            int(
                self.camera_max_observed_lag
            ),
            int(lag),
        )

        should_log = (
            self.camera_lag_event_count
            <= CAMERA_LAG_LOG_FIRST_N
            or
            self.camera_lag_event_count
            % CAMERA_LAG_LOG_EVERY
            == 0
            or
            lag
            >= CAMERA_MAX_AGE_TICKS
        )

        if should_log:
            print(
                "CAMERA LAG | event={} | sim_frame={} | camera_frame={} | "
                "lag={} | max_seen={}".format(
                    self.camera_lag_event_count,
                    sim_frame,
                    camera_frame,
                    lag,
                    self.camera_max_observed_lag,
                )
            )

    def _wait_callbacks_for_frame(
        self,
        frame,
        timeout_seconds=(
            SENSOR_CALLBACK_TIMEOUT_S
        ),
    ):
        start = time.time()

        while True:
            camera_frame = (
                int(
                    getattr(
                        self.camera_obj,
                        "frame",
                        -1,
                    )
                )
                if self.camera_obj is not None
                else -1
            )

            camera_age_ticks = (
                int(frame)
                - int(camera_frame)
                if camera_frame >= 0
                else 10 ** 9
            )

            camera_ready = bool(
                self.camera_obj is not None
                and bool(
                    getattr(
                        self.camera_obj,
                        "ready",
                        False,
                    )
                )
                and len(
                    self.camera_obj.front_camera
                )
                > 0
                and camera_frame >= 0
                and camera_age_ticks
                <= CAMERA_MAX_AGE_TICKS
            )

            imu_frame = (
                int(
                    getattr(
                        self.imu_obj,
                        "frame",
                        -1,
                    )
                )
                if self.imu_obj is not None
                else -1
            )

            imu_ready = bool(
                self.imu_obj is not None
                and bool(
                    getattr(
                        self.imu_obj,
                        "ready",
                        False,
                    )
                )
                and imu_frame >= int(frame)
            )

            if (
                camera_ready
                and imu_ready
            ):
                self.last_camera_frame = int(
                    camera_frame
                )

                self.last_camera_frame_lag = int(
                    camera_age_ticks
                )

                self._log_camera_lag(
                    sim_frame=int(frame),
                    camera_frame=int(
                        camera_frame
                    ),
                    lag=int(
                        camera_age_ticks
                    ),
                )

                return

            if (
                time.time() - start
                > timeout_seconds
            ):
                raise TimeoutError(
                    "Sensor timeout | sim_frame={} | "
                    "camera_ready={} | camera_frame={} | "
                    "camera_age_ticks={} | max_age_ticks={} | "
                    "imu_ready={} | imu_frame={}".format(
                        int(frame),
                        bool(camera_ready),
                        int(camera_frame),
                        int(camera_age_ticks),
                        int(CAMERA_MAX_AGE_TICKS),
                        bool(imu_ready),
                        int(imu_frame),
                    )
                )

            time.sleep(0.001)

    def _tick_and_wait_sensors(self):
        frame = int(
            self.world.tick()
        )

        self.last_sim_frame = frame

        self._wait_callbacks_for_frame(
            frame
        )

        return frame

    def _prime_sensors(self):
        last_error = None

        for _ in range(
            SENSOR_PRIME_MAX_TICKS
        ):
            try:
                self._tick_and_wait_sensors()
                return

            except TimeoutError as error:
                last_error = error

        raise TimeoutError(
            "Không prime được camera/IMU: {}".format(
                last_error
            )
        )

    # ==================================================================
    # Observation
    # ==================================================================

    def _build_observation(self):
        if (
            self.camera_obj is None
            or not self.camera_obj.front_camera
        ):
            raise RuntimeError(
                "Camera chưa có RGB frame."
            )

        clean_imu_state = (
            self.imu_obj.get_state()
        )

        if not clean_imu_state[
            "imu_ready"
        ]:
            raise RuntimeError(
                "IMU chưa ready."
            )

        raw_rgb = (
            self.camera_obj
            .front_camera[-1]
        )

        camera_frame = int(
            getattr(
                self.camera_obj,
                "frame",
                -1,
            )
        )

        # Camera augmentation is tied to CAMERA FRAME, not 50-Hz tick.
        if self.vision_dr_enabled and self._episode_dr_active:
            if (
                self.last_augmented_rgb is None
                or
                self._last_augmented_camera_frame
                != camera_frame
            ):
                augmented_rgb = (
                    augment_camera_rgb_v5(
                        image_rgb=raw_rgb,
                        params=(
                            self.vision_sensor_params
                        ),
                        np_rng=(
                            self._vision_np_rng
                        ),
                    )
                )

                self.last_augmented_rgb = (
                    augmented_rgb
                )

                self._last_augmented_camera_frame = (
                    camera_frame
                )
            else:
                augmented_rgb = (
                    self.last_augmented_rgb
                )
        else:
            augmented_rgb = raw_rgb

        if self.sensor_dr_enabled and self._episode_dr_active:
            policy_state = (
                noisy_state_for_policy_v5(
                    clean_imu_state=(
                        clean_imu_state
                    ),
                    params=(
                        self.vision_sensor_params
                    ),
                    np_rng=(
                        self._vision_np_rng
                    ),
                )
            )
        else:
            policy_state = {
                "speed_mps": float(
                    clean_imu_state[
                        "speed_mps"
                    ]
                ),
                "yaw_rate_rad_s": float(
                    clean_imu_state[
                        "yaw_rate_rad_s"
                    ]
                ),
                "longitudinal_accel_mps2": float(
                    clean_imu_state[
                        "longitudinal_accel_mps2"
                    ]
                ),
            }

        observation = (
            self.observation_builder.build(
                image_rgb=augmented_rgb,
                speed_mps=(
                    policy_state[
                        "speed_mps"
                    ]
                ),
                yaw_rate_rad_s=(
                    policy_state[
                        "yaw_rate_rad_s"
                    ]
                ),
                longitudinal_accel_mps2=(
                    policy_state[
                        "longitudinal_accel_mps2"
                    ]
                ),
                prev_steer_cmd=(
                    self.action_controller
                    .prev_steer_cmd
                ),
                prev_speed_cmd_mps=(
                    self.action_controller
                    .prev_speed_cmd_mps
                ),
            )
        )

        self.last_policy_sensor_state = dict(
            policy_state
        )

        # Reward receives CLEAN state.
        return (
            observation,
            clean_imu_state,
        )

    # ==================================================================
    # Reset / step
    # ==================================================================

    def reset(self):
        try:
            self.reset_count += 1

            # Sample one domain mode per episode. All DR families are enabled
            # from the beginning of training, but nominal episodes are mixed
            # in deliberately to prevent permanent noise-compensation habits.
            any_dr = bool(
                self.dynamics_dr_enabled
                or self.vision_dr_enabled
                or self.sensor_dr_enabled
            )
            self._episode_dr_active = bool(
                any_dr
                and self._domain_rng.random() < self.dr_episode_probability
            )

            self._sample_recovery_scenario()
            self._sample_vision_sensor_episode()

            self.destroy_episode_actors()
            self._drain_destroyed_actors()
            self._debug_actor_counts()

            self._apply_weather()

            self.vehicle = (
                self._spawn_vehicle()
            )

            self._apply_vehicle_geometry()
            self._prepare_episode_dynamics_dr()
            self._settle_vehicle()
            self._spawn_episode_components()
            self._prime_sensors()

            self._debug_actor_counts()

            # Reward progress starts only after setup ticks.
            self.reward_manager.reset(
                prev_steer_cmd=0.0,
                prev_speed_cmd_mps=0.0,
            )

            (
                observation,
                clean_imu_state,
            ) = self._build_observation()

            self.episode_done = False

            print(
                "RESET V5 OK | episode={} | spawn={} | "
                "scenario=NOMINAL_BALANCED_SPAWN | offset={:+.3f}m | heading={:+.1f}deg | "
                "domain={} | recovery_selected={} min_step={} stable_need={} | "
                "obs={} | speed={:.3f}m/s".format(
                    self._episode_index,
                    self.current_spawn_index + 1,
                    float(self._spawn_offset_m),
                    float(self._spawn_heading_deg),
                    "FULL_DR" if self._episode_dr_active else "NOMINAL",
                    bool(self._episode_selected_for_disturbance),
                    int(RECOVERY_CONFIG_V5.recovery_min_episode_step),
                    int(RECOVERY_CONFIG_V5.recovery_stable_ticks),
                    tuple(observation.shape),
                    float(clean_imu_state["speed_mps"]),
                )
            )

            return observation

        except Exception:
            print(
                "\n===== V5 RESET ERROR ====="
            )
            traceback.print_exc()
            print(
                "==========================\n"
            )

            self.destroy_episode_actors()
            raise

    def step(self, action):
        """
        ONE V5 STEP PATH.

        1. PPO action
        2. optional recovery disturbance
        3. action controller / command delay / steer gain
        4. CARLA tick
        5. obs100
        6. RewardManagerV5 computes ALL reward terms
        7. return final reward
        """
        if (
            self.vehicle is None
            or self.episode_done
        ):
            raise RuntimeError(
                "Episode chưa reset hoặc đã done. Hãy gọi reset()."
            )

        try:
            self._episode_step += 1

            ppo_action = (
                _to_action_array(
                    action
                )
            )

            requested_steer = float(
                np.clip(
                    ppo_action[0],
                    -1.0,
                    1.0,
                )
            )

            requested_speed = float(
                np.clip(
                    ppo_action[1],
                    0.0,
                    float(
                        REAL.max_speed_mps
                    ),
                )
            )

            applied_steer = float(
                requested_steer
            )

            disturbance_active = bool(
                self._disturbance_enabled
                and
                self._disturbance_start_tick
                <= self._episode_step
                < self._disturbance_end_tick
            )

            if disturbance_active:
                applied_steer = _clip(
                    applied_steer
                    + float(
                        self._disturbance_steer
                    ),
                    -1.0,
                    1.0,
                )

            control_info = (
                self.action_controller.step(
                    steer_cmd=(
                        applied_steer
                    ),
                    speed_cmd_mps=(
                        requested_speed
                    ),
                    dt=self.dt,
                )
            )

            sim_frame = (
                self._tick_and_wait_sensors()
            )

            (
                observation,
                clean_imu_state,
            ) = self._build_observation()

            # Decide whether recovery may start on the NEXT control tick.
            self._update_recovery_curriculum_gate(
                clean_imu_state
            )

            # ----------------------------------------------------------
            # SINGLE REWARD PATH
            # ----------------------------------------------------------
            (
                reward,
                done,
                info,
            ) = self.reward_manager.step(
                speed_mps=(
                    clean_imu_state[
                        "speed_mps"
                    ]
                ),
                steer_cmd=(
                    control_info[
                        "actual_steer_cmd"
                    ]
                ),
                speed_cmd_mps=(
                    control_info[
                        "actual_speed_cmd_mps"
                    ]
                ),
                desired_speed_mps=(
                    self.desired_speed_mps
                ),
                dt=self.dt,
            )

            # ----------------------------------------------------------
            # Environment diagnostics only.
            # NO reward modification below this point.
            # ----------------------------------------------------------
            info["sim_frame"] = int(
                sim_frame
            )

            info["camera_frame"] = int(
                self.last_camera_frame
            )

            info["camera_frame_lag"] = (
                int(
                    self.last_camera_frame_lag
                )
                if self.last_camera_frame_lag
                is not None
                else None
            )

            info["control"] = dict(
                control_info
            )

            info["imu_clean"] = dict(
                clean_imu_state
            )

            if (
                self.last_policy_sensor_state
                is not None
            ):
                info[
                    "policy_sensor_state"
                ] = dict(
                    self.last_policy_sensor_state
                )

            info[
                "domain_randomization"
            ] = dict(
                self.domain_randomization_params
            )

            info[
                "vision_sensor_dr"
            ] = dict(
                self.vision_sensor_params
            )

            info[
                "dynamics_dr_enabled"
            ] = bool(
                self.dynamics_dr_enabled and self._episode_dr_active
            )

            info[
                "vision_dr_enabled"
            ] = bool(
                self.vision_dr_enabled and self._episode_dr_active
            )

            info[
                "sensor_dr_enabled"
            ] = bool(
                self.sensor_dr_enabled and self._episode_dr_active
            )

            info["dr_episode_active"] = bool(self._episode_dr_active)
            info["dr_episode_probability"] = float(self.dr_episode_probability)
            info["dynamics_dr_configured"] = bool(self.dynamics_dr_enabled)
            info["vision_dr_configured"] = bool(self.vision_dr_enabled)
            info["sensor_dr_configured"] = bool(self.sensor_dr_enabled)

            info[
                "recovery_scenarios_enabled"
            ] = bool(
                self.recovery_scenarios_enabled
            )

            info[
                "spawn_recovery"
            ] = bool(
                self._spawn_recovery
            )

            info["spawn_number"] = (
                int(self.current_spawn_index) + 1
                if self.current_spawn_index is not None
                else None
            )

            info[
                "spawn_offset_m"
            ] = float(
                self._spawn_offset_m
            )

            info[
                "spawn_heading_deg"
            ] = float(
                self._spawn_heading_deg
            )

            info[
                "ppo_action"
            ] = (
                float(
                    requested_steer
                ),
                float(
                    requested_speed
                ),
            )

            info[
                "applied_action"
            ] = (
                float(
                    applied_steer
                ),
                float(
                    requested_speed
                ),
            )

            info[
                "disturbance_active"
            ] = bool(
                disturbance_active
            )

            # IMPORTANT: expose the sampled disturbance steer on the TRIGGER
            # step as well as on active disturbance ticks.  The trigger is
            # decided after the current tick's `disturbance_active` flag was
            # computed, so gating this field on `disturbance_active` made the
            # trigger event carry steer=0.0.  VNext's recovery audit therefore
            # could not determine + / - injection sign and Stage1 could never
            # satisfy its recovery gate.
            info[
                "disturbance_steer"
            ] = float(self._disturbance_steer)

            info["disturbance_trigger_event"] = bool(
                self._disturbance_trigger_event
            )
            info["disturbance_event_count"] = int(
                self._disturbance_event_count
            )
            info["disturbance_success_event"] = bool(
                self._disturbance_success_event
            )
            info["disturbance_recovery_pending"] = bool(
                self._disturbance_recovery_pending
            )
            info["disturbance_recovery_success_hold"] = int(
                self._disturbance_recovery_success_hold
            )
            info["recovery_curriculum_allowed"] = bool(
                self.recovery_scenarios_enabled
                and self._episode_step
                >= int(RECOVERY_CONFIG_V5.recovery_min_episode_step)
            )
            info["recovery_stable_gate_ticks"] = int(
                self._stable_gate_ticks
            )
            info["recovery_min_episode_step"] = int(
                RECOVERY_CONFIG_V5.recovery_min_episode_step
            )
            info["recovery_max_events_per_episode"] = int(
                RECOVERY_CONFIG_V5.recovery_max_events_per_episode
            )

            info[
                "observation_shape"
            ] = tuple(
                observation.shape
            )

            info[
                "observation_device"
            ] = str(
                observation.device
            )

            self.episode_done = bool(
                done
            )

            return (
                observation,
                float(reward),
                bool(done),
                info,
            )

        except Exception:
            print(
                "\n===== V5 STEP ERROR ====="
            )
            traceback.print_exc()
            print(
                "=========================\n"
            )

            self.episode_done = True
            raise

    # ==================================================================
    # Cleanup
    # ==================================================================

    def destroy_episode_actors(self):
        if self.reward_manager is not None:
            try:
                self.reward_manager.destroy()
            except Exception:
                pass

            self.reward_manager = None

        for sensor_wrapper_name in (
            "camera_obj",
            "imu_obj",
        ):
            wrapper = getattr(
                self,
                sensor_wrapper_name,
                None,
            )

            if wrapper is not None:
                sensor = getattr(
                    wrapper,
                    "sensor",
                    None,
                )

                if sensor is not None:
                    try:
                        sensor.stop()
                    except Exception:
                        pass

                    try:
                        if sensor.is_alive:
                            sensor.destroy()
                    except Exception:
                        try:
                            sensor.destroy()
                        except Exception:
                            pass

            setattr(
                self,
                sensor_wrapper_name,
                None,
            )

        if self.vehicle is not None:
            try:
                self.vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        steer=0.0,
                        brake=1.0,
                        hand_brake=True,
                    )
                )
            except Exception:
                pass

            try:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except Exception:
                pass

        self.vehicle = None
        self.action_controller = None
        self.episode_done = True

    def close(self):
        self.destroy_episode_actors()

        if self._initial_weather is not None:
            try:
                self.world.set_weather(
                    self._initial_weather
                )
            except Exception:
                pass

        try:
            self.world.apply_settings(
                self._original_world_settings
            )
        except Exception:
            pass

    cleanup = close


# Canonical imports used by the guarded multi-map trainer/evaluator.
CarlaEnvironmentRGB = CarlaEnvironmentRGBV5
CarlaEnvironmentRGBV5Simple = CarlaEnvironmentRGBV5
CarlaEnvironmentRGBV5MultiMap = CarlaEnvironmentRGBV5
