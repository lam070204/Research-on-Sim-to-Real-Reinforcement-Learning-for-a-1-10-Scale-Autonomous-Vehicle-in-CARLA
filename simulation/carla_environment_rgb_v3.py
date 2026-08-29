# -*- coding: utf-8 -*-
"""
GĐ13 - Clean CARLA RGB V3 environment.

Policy contract
---------------
Observation:
    torch.float32, shape (100,)
    [latent95, speed, yaw_rate, accel_x, prev_steer, prev_speed_cmd]

Action:
    [steer_cmd, speed_cmd_mps]
    steer_cmd       in [-1, +1]
    speed_cmd_mps   in [0, 1] real-equivalent m/s

Reward-only privileged CARLA state is NOT added to observation.

The environment owns:
- synchronous CARLA stepping at 50 Hz
- vehicle spawn + measured wheel geometry
- camera + IMU
- frozen RGB VAE observation builder
- ActionControllerV3
- RewardManagerV3
- episode cleanup

Removed from the old V2-style environment:
- throttle as PPO action
- route/navigation observation
- lateral/heading in policy observation
- pygame display
- pedestrians/NPCs
- old reward logic
- old route bookkeeping/checkpoint logic
"""

import math
import random
import time
import traceback

import numpy as np

from simulation.carla_connection_v3 import carla
from simulation.carla_sensors_v3 import CameraSensor, IMUSensorV3

from action_controller_v3 import ActionControllerV3

# GĐ6_DOMAIN_RANDOMIZATION_INTEGRATED
from domain_randomization_v3 import (
    nominal_domain_parameters,
    sample_domain_randomization,
)
from domain_randomization_runtime_v3 import (
    apply_domain_physics_v3,
    DomainRandomizedActionControllerV3,
)
from encoder_runtime_rgb_v3 import EncodeRGBV3
from observation_builder_rgb_v3 import (
    ObservationBuilderRGBV3,
    OBSERVATION_DIM_V3,
)
from reward_manager_v3 import (
    RewardManagerV3,
    EpisodeConfigV3,
)
from vehicle_specs_v3 import (
    VEHICLE_BLUEPRINT_ID,
    REAL,
    CARLA_WHEEL_RADIUS_CM,
    CARLA_FRONT_MAX_STEER_DEG,
    CARLA_REAR_MAX_STEER_DEG,
    CONTROL_HZ,
    CONTROL_DT,
    FRONT_CAMERA_FPS,
)


CONTROL_HZ = 50.0
CONTROL_DT = 1.0 / CONTROL_HZ

DEFAULT_DESIRED_SPEED_MPS = 0.60
DEFAULT_MAX_EPISODE_SECONDS = 60.0

DEFAULT_SPAWN_Z_OFFSET_M = 0.25
DEFAULT_SETTLE_SECONDS = 3.0

SENSOR_CALLBACK_TIMEOUT_S = 3.0
CAMERA_MAX_AGE_S = 0.080
CAMERA_MAX_AGE_TICKS = int(math.ceil(CAMERA_MAX_AGE_S / CONTROL_DT))
SENSOR_PRIME_MAX_TICKS = 50
DESTROY_DRAIN_TICKS = 2
ACTOR_DEBUG_EVERY_RESETS = 25

# CARLA RGB cameras are GPU-backed and can arrive 1-2 simulation frames late.
# Accept a recent camera frame instead of requiring exact equality with world.tick().
CAMERA_MAX_FRAME_LAG = 2
CAMERA_LAG_LOG_FIRST_N = 10
CAMERA_LAG_LOG_EVERY = 100


def _to_action_array(action):
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()

    array = np.asarray(action, dtype=np.float32).reshape(-1)

    if array.shape != (2,):
        raise ValueError(
            "Action V3 phải có shape (2,), nhận được {}.".format(
                tuple(array.shape)
            )
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Action V3 chứa NaN/Inf: {}".format(array)
        )

    return array


class CarlaEnvironmentRGB:
    """
    Canonical V3 training environment.

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
        domain_randomization_enabled=False,
        domain_randomization_seed=None,
    ):
        self.client = client
        self.world = world
        self.town = town

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        self.control_hz = float(control_hz)
        if self.control_hz <= 0.0:
            raise ValueError("control_hz phải > 0.")
        self.dt = 1.0 / self.control_hz

        # Control/state = 50 Hz; camera = 30 FPS. Reuse latest camera frame giữa các control tick.
        if abs(self.control_hz - 50.0) > 1e-6:
            raise ValueError(
                "V3 khóa control_hz = 50 Hz để khớp vòng điều khiển xe thật."
            )

        self.desired_speed_mps = self._sanitize_desired_speed(
            desired_speed_mps
        )
        self.max_episode_seconds = float(max_episode_seconds)

        self.spawn_z_offset_m = float(spawn_z_offset_m)
        self.settle_seconds = float(settle_seconds)

        self.steer_rate_limit_per_s = steer_rate_limit_per_s
        self.speed_rate_limit_mps2 = speed_rate_limit_mps2

        self.domain_randomization_enabled = bool(
            domain_randomization_enabled
        )
        self.domain_randomization_seed = domain_randomization_seed
        self._domain_rng = random.Random(domain_randomization_seed)
        self.domain_randomization_params = nominal_domain_parameters()
        self.domain_randomization_applied = None

        self._original_world_settings = self.world.get_settings()
        self._configure_synchronous_world()

        self.spawn_points = list(self.map.get_spawn_points())
        if not self.spawn_points:
            raise RuntimeError("Map hiện tại không có spawn point.")

        self.safe_spawn_indices = self._resolve_spawn_indices(
            safe_spawn_numbers
        )
        self.spawn_order = list(self.safe_spawn_indices)
        random.shuffle(self.spawn_order)
        self.spawn_cursor = 0
        self.current_spawn_index = None

        # One frozen encoder reused across all episodes.
        encoder = EncodeRGBV3(device=encoder_device)
        self.observation_builder = ObservationBuilderRGBV3(
            encoder=encoder
        )

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
            "ENV RGB V3 CLEAN | obs={} | action=[steer, speed] | "
            "Hz={:.1f} | desired_speed={:.3f} m/s | spawns={}".format(
                OBSERVATION_DIM_V3,
                self.control_hz,
                self.desired_speed_mps,
                [i + 1 for i in self.safe_spawn_indices],
            )
        )

    def _sanitize_desired_speed(self, value):
        value = float(value)
        max_speed = float(REAL.max_speed_mps)

        if not np.isfinite(value):
            raise ValueError("desired_speed_mps phải finite.")

        return float(np.clip(value, 0.0, max_speed))

    def set_desired_speed_mps(self, value):
        """
        Curriculum hook for GĐ15.
        Does not change policy action bounds.
        """
        self.desired_speed_mps = self._sanitize_desired_speed(value)

    def _configure_synchronous_world(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)
        self.last_sim_frame = int(self.world.tick())

    def _resolve_spawn_indices(self, safe_spawn_numbers):
        if safe_spawn_numbers is None:
            return list(range(len(self.spawn_points)))

        indices = []
        for spawn_number in safe_spawn_numbers:
            index = int(spawn_number) - 1
            if 0 <= index < len(self.spawn_points):
                indices.append(index)
            else:
                print(
                    "WARNING: bỏ spawn {} vì map chỉ có {} spawn.".format(
                        spawn_number,
                        len(self.spawn_points),
                    )
                )

        # Preserve order but remove duplicates.
        indices = list(dict.fromkeys(indices))

        if not indices:
            raise RuntimeError("safe_spawn_numbers không có spawn hợp lệ.")

        return indices

    def _next_spawn_transform(self):
        if self.spawn_cursor >= len(self.spawn_order):
            self.spawn_order = list(self.safe_spawn_indices)
            random.shuffle(self.spawn_order)
            self.spawn_cursor = 0

        index = self.spawn_order[self.spawn_cursor]
        self.spawn_cursor += 1
        self.current_spawn_index = index

        original = self.spawn_points[index]

        return carla.Transform(
            carla.Location(
                x=original.location.x,
                y=original.location.y,
                z=original.location.z + self.spawn_z_offset_m,
            ),
            original.rotation,
        )

    def _get_vehicle_blueprint(self):
        matches = self.blueprint_library.filter(
            VEHICLE_BLUEPRINT_ID
        )

        if not matches:
            raise RuntimeError(
                "Không tìm thấy blueprint {}.".format(
                    VEHICLE_BLUEPRINT_ID
                )
            )

        return matches[0]

    def _spawn_vehicle(self):
        vehicle_bp = self._get_vehicle_blueprint()

        for _ in range(len(self.safe_spawn_indices)):
            transform = self._next_spawn_transform()

            vehicle = self.world.try_spawn_actor(
                vehicle_bp,
                transform,
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
            raise RuntimeError("Vehicle chưa được spawn.")

        if self.vehicle.type_id != VEHICLE_BLUEPRINT_ID:
            raise RuntimeError(
                "Sai blueprint: {} != {}".format(
                    self.vehicle.type_id,
                    VEHICLE_BLUEPRINT_ID,
                )
            )

        physics = self.vehicle.get_physics_control()
        wheels = list(physics.wheels)

        if len(wheels) < 4:
            raise RuntimeError(
                "Vehicle trả về {} wheels, cần 4.".format(
                    len(wheels)
                )
            )

        for wheel in wheels[:4]:
            wheel.radius = float(CARLA_WHEEL_RADIUS_CM)

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
        self.vehicle.apply_physics_control(physics)

        # Let CARLA commit the physics update.
        self.last_sim_frame = int(self.world.tick())

        applied = list(
            self.vehicle.get_physics_control().wheels
        )

        radius_ok = all(
            abs(float(applied[i].radius) - CARLA_WHEEL_RADIUS_CM) < 1e-3
            for i in range(4)
        )
        steer_ok = (
            abs(
                float(applied[0].max_steer_angle)
                - CARLA_FRONT_MAX_STEER_DEG
            ) < 1e-3
            and abs(
                float(applied[1].max_steer_angle)
                - CARLA_FRONT_MAX_STEER_DEG
            ) < 1e-3
            and abs(
                float(applied[2].max_steer_angle)
                - CARLA_REAR_MAX_STEER_DEG
            ) < 1e-3
            and abs(
                float(applied[3].max_steer_angle)
                - CARLA_REAR_MAX_STEER_DEG
            ) < 1e-3
        )

        if not radius_ok or not steer_ok:
            raise RuntimeError(
                "Wheel radius/steering không giữ đúng sau apply."
            )

    def _prepare_episode_domain_v3(self):
        if self.domain_randomization_enabled:
            params = sample_domain_randomization(
                rng=self._domain_rng
            )
        else:
            params = nominal_domain_parameters()

        self.domain_randomization_params = dict(params)

        self.domain_randomization_applied = apply_domain_physics_v3(
            vehicle=self.vehicle,
            params=self.domain_randomization_params,
            carla=carla,
        )

        self.last_sim_frame = int(self.world.tick())

        verify = self.vehicle.get_physics_control()

        expected_brake = float(
            self.domain_randomization_params["max_brake_torque"]
        )
        applied_wheels = list(verify.wheels)

        brake_ok = all(
            abs(float(w.max_brake_torque) - expected_brake) < 1e-3
            for w in applied_wheels
        )

        autobox_ok = (
            bool(verify.use_gear_autobox)
            == bool(
                self.domain_randomization_params[
                    "use_gear_autobox"
                ]
            )
        )

        gear_switch_ok = (
            abs(
                float(verify.gear_switch_time)
                - float(
                    self.domain_randomization_params[
                        "gear_switch_time_s"
                    ]
                )
            )
            < 1e-6
        )

        if not brake_ok or not autobox_ok or not gear_switch_ok:
            raise RuntimeError(
                "GĐ6 PhysicsControl readback mismatch."
            )

        print(
            "GĐ6 DOMAIN | random={} | torque_scale={:.4f} | "
            "brake={:.1f} | steer_gain(+/-)={:.4f}/{:.4f} | "
            "delay={} tick".format(
                self.domain_randomization_enabled,
                float(
                    self.domain_randomization_params[
                        "torque_scale"
                    ]
                ),
                float(
                    self.domain_randomization_params[
                        "max_brake_torque"
                    ]
                ),
                float(
                    self.domain_randomization_params[
                        "steer_gain_positive"
                    ]
                ),
                float(
                    self.domain_randomization_params[
                        "steer_gain_negative"
                    ]
                ),
                int(
                    self.domain_randomization_params[
                        "extra_command_delay_ticks"
                    ]
                ),
            )
        )

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
            carla.Vector3D(0.0, 0.0, 0.0)
        )
        self.vehicle.set_target_angular_velocity(
            carla.Vector3D(0.0, 0.0, 0.0)
        )

        n_ticks = max(
            1,
            int(round(self.settle_seconds * self.control_hz)),
        )

        for _ in range(n_ticks):
            self.last_sim_frame = int(self.world.tick())

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
            )
        )

    def _spawn_episode_components(self):
        self.camera_obj = CameraSensor(self.vehicle)
        self.imu_obj = IMUSensorV3(self.vehicle)

        base_action_controller = ActionControllerV3(
            vehicle=self.vehicle,
            steer_rate_limit_per_s=self.steer_rate_limit_per_s,
            speed_rate_limit_mps2=self.speed_rate_limit_mps2,
        )

        self.action_controller = DomainRandomizedActionControllerV3(
            base_controller=base_action_controller,
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
        self.action_controller.reset()

        self.reward_manager = RewardManagerV3(
            vehicle=self.vehicle,
            episode_config=EpisodeConfigV3(
                max_episode_seconds=self.max_episode_seconds,
                terminate_on_collision=True,
                terminate_on_offroad=True,
                terminate_on_stuck=True,
            ),
        )

    def _drain_destroyed_actors(self):
        """
        Advance a couple of synchronous ticks after actor destruction.

        CARLA actor.destroy() removes the actor, but GPU/render resources
        associated with camera actors may be released asynchronously by
        Unreal. Giving the server a small gap before spawning the next
        episode camera reduces destroy/spawn churn on the render thread.
        """
        for _ in range(DESTROY_DRAIN_TICKS):
            self.last_sim_frame = int(self.world.tick())

    def _debug_actor_counts(self, force=False):
        if (
            not force
            and self.reset_count % ACTOR_DEBUG_EVERY_RESETS != 0
        ):
            return

        try:
            actors = self.world.get_actors()
            vehicles = list(actors.filter("vehicle.*"))
            sensors = list(actors.filter("sensor.*"))

            sensor_types = {}
            for actor in sensors:
                sensor_types[actor.type_id] = (
                    sensor_types.get(actor.type_id, 0) + 1
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

    def _log_camera_lag(self, sim_frame, camera_frame, lag):
        if lag <= 0:
            return

        self.camera_lag_event_count += 1
        self.camera_max_observed_lag = max(
            int(self.camera_max_observed_lag),
            int(lag),
        )

        should_log = (
            self.camera_lag_event_count <= CAMERA_LAG_LOG_FIRST_N
            or self.camera_lag_event_count % CAMERA_LAG_LOG_EVERY == 0
            or lag >= CAMERA_MAX_FRAME_LAG
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
        timeout_seconds=SENSOR_CALLBACK_TIMEOUT_S,
    ):
        start = time.time()

        while True:
            camera_frame = (
                int(getattr(self.camera_obj, "frame", -1))
                if self.camera_obj is not None
                else -1
            )

            camera_age_ticks = (
                int(frame) - camera_frame
                if camera_frame >= 0
                else 10 ** 9
            )

            camera_ready = (
                self.camera_obj is not None
                and bool(getattr(self.camera_obj, "ready", False))
                and len(self.camera_obj.front_camera) > 0
                and camera_frame >= 0
                and camera_age_ticks <= CAMERA_MAX_AGE_TICKS
            )

            imu_frame = (
                int(getattr(self.imu_obj, "frame", -1))
                if self.imu_obj is not None
                else -1
            )

            imu_ready = (
                self.imu_obj is not None
                and bool(getattr(self.imu_obj, "ready", False))
                and imu_frame >= int(frame)
            )

            if camera_ready and imu_ready:
                return

            if time.time() - start > timeout_seconds:
                raise TimeoutError(
                    "Sensor timeout | sim_frame={} | "
                    "camera_ready={} | camera_frame={} | camera_age_ticks={} | "
                    "camera_max_age_ticks={} | "
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
        # Một control tick = 20 ms.
        # Không clear camera buffer:
        # camera thật chỉ 30 FPS nên nhiều control tick có thể dùng cùng RGB frame.
        frame = int(self.world.tick())
        self.last_sim_frame = frame

        self._wait_callbacks_for_frame(frame)

        return frame

    def _prime_sensors(self):
        last_error = None

        for _ in range(SENSOR_PRIME_MAX_TICKS):
            try:
                self._tick_and_wait_sensors()
                return
            except TimeoutError as error:
                last_error = error

        raise TimeoutError(
            "Không prime được camera/IMU: {}".format(last_error)
        )

    def _build_observation(self):
        if not self.camera_obj.front_camera:
            raise RuntimeError("Camera chưa có RGB frame.")

        imu_state = self.imu_obj.get_state()

        if not imu_state["imu_ready"]:
            raise RuntimeError("IMU chưa ready.")

        image_rgb = self.camera_obj.front_camera[-1]

        observation = self.observation_builder.build(
            image_rgb=image_rgb,
            speed_mps=imu_state["speed_mps"],
            yaw_rate_rad_s=imu_state["yaw_rate_rad_s"],
            longitudinal_accel_mps2=imu_state[
                "longitudinal_accel_mps2"
            ],
            prev_steer_cmd=self.action_controller.prev_steer_cmd,
            prev_speed_cmd_mps=(
                self.action_controller.prev_speed_cmd_mps
            ),
        )

        return observation, imu_state

    def reset(self):
        """
        Start a fresh episode and return obs(100,).
        """
        try:
            self.reset_count += 1

            self.destroy_episode_actors()
            self._drain_destroyed_actors()
            self._debug_actor_counts()

            self.vehicle = self._spawn_vehicle()

            self._apply_vehicle_geometry()
            self._prepare_episode_domain_v3()
            self._settle_vehicle()
            self._spawn_episode_components()
            self._prime_sensors()
            self._debug_actor_counts()

            # Prime ticks are setup only. Reward/progress starts here.
            self.reward_manager.reset(
                prev_steer_cmd=0.0,
                prev_speed_cmd_mps=0.0,
            )

            observation, imu_state = self._build_observation()

            self.episode_done = False

            print(
                "RESET OK | spawn={} | obs={} | device={} | "
                "speed={:.3f} m/s".format(
                    self.current_spawn_index + 1,
                    tuple(observation.shape),
                    observation.device,
                    imu_state["speed_mps"],
                )
            )

            return observation

        except Exception:
            print("\n===== GĐ13 RESET ERROR =====")
            traceback.print_exc()
            print("============================\n")
            self.destroy_episode_actors()
            raise

    def step(self, action):
        """
        action = [steer_cmd, speed_cmd_mps]

        Returns:
            obs, reward, done, info
        """
        if self.vehicle is None or self.episode_done:
            raise RuntimeError(
                "Episode chưa reset hoặc đã done. Hãy gọi reset()."
            )

        try:
            action_array = _to_action_array(action)

            requested_steer = float(
                np.clip(action_array[0], -1.0, 1.0)
            )
            requested_speed = float(
                np.clip(
                    action_array[1],
                    0.0,
                    float(REAL.max_speed_mps),
                )
            )

            control_info = self.action_controller.step(
                steer_cmd=requested_steer,
                speed_cmd_mps=requested_speed,
                dt=self.dt,
            )

            sim_frame = self._tick_and_wait_sensors()

            observation, imu_state = self._build_observation()

            reward, done, info = self.reward_manager.step(
                speed_mps=imu_state["speed_mps"],
                steer_cmd=control_info["actual_steer_cmd"],
                speed_cmd_mps=control_info[
                    "actual_speed_cmd_mps"
                ],
                desired_speed_mps=self.desired_speed_mps,
                dt=self.dt,
            )

            info["sim_frame"] = int(sim_frame)
            info["camera_frame"] = int(self.last_camera_frame)
            info["camera_frame_lag"] = (
                int(self.last_camera_frame_lag)
                if self.last_camera_frame_lag is not None
                else None
            )
            info["camera_max_frame_lag_allowed"] = int(CAMERA_MAX_FRAME_LAG)
            info["control"] = control_info
            info["imu"] = imu_state
            info["domain_randomization"] = dict(
                self.domain_randomization_params
            )
            info["observation_shape"] = tuple(observation.shape)
            info["observation_device"] = str(observation.device)

            self.episode_done = bool(done)

            return observation, float(reward), bool(done), info

        except Exception:
            print("\n===== GĐ13 STEP ERROR =====")
            traceback.print_exc()
            print("===========================\n")
            self.episode_done = True
            raise

    def destroy_episode_actors(self):
        """
        Destroy only episode-owned objects.
        World settings are not restored here because reset() needs sync mode.
        """
        if self.reward_manager is not None:
            try:
                self.reward_manager.destroy()
            except Exception:
                pass
            self.reward_manager = None

        for sensor_wrapper_name in ("camera_obj", "imu_obj"):
            wrapper = getattr(self, sensor_wrapper_name, None)

            if wrapper is not None:
                sensor = getattr(wrapper, "sensor", None)

                if sensor is not None:
                    try:
                        sensor.stop()
                    except Exception:
                        pass

                    try:
                        if sensor.is_alive:
                            sensor.destroy()
                    except Exception:
                        pass

            setattr(self, sensor_wrapper_name, None)

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

        try:
            self.world.apply_settings(
                self._original_world_settings
            )
        except Exception:
            pass

    # Compatibility alias for code that uses cleanup().
    cleanup = close


# Explicit V3 alias for new code.
CarlaEnvironmentRGBV3 = CarlaEnvironmentRGB
