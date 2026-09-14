# -*- coding: utf-8 -*-
"""
Evaluate / load ONE exact PPO V3 checkpoint.

- Does not modify frozen train_ppo_rgb_v3.py.
- Loads an exact .pth path with PPOAgent.load(checkpoint_path=...).
- Runs deterministic evaluation: agent.get_action(obs, train=False).
- Domain randomization OFF during evaluation.

Example:
python .\evaluate_specific_checkpoint_v3_monitor.py `
  --checkpoint ".\preTrained_models\PPO\automav3_rgb_v3\ppo_policy_40_.pth" `
  --desired-speed 1.0 `
  --test-timesteps 5000 `
  --monitor ppo `
  --print-every-steps 50


  python .\evaluate_specific_checkpoint_v3_monitor.py `
  --checkpoint ".\preTrained_models\PPO\automav3_rgb_v3\ppo_policy_40_.pth" `
  --desired-speed 1.0 `
  --test-timesteps 5000 `
  --monitor third `
  --print-every-steps 50


  python .\evaluate_specific_checkpoint_v3_monitor.py `
  --checkpoint ".\preTrained_models\PPO\automav3_rgb_v3\ppo_policy_40_.pth" `
  --desired-speed 1.0 `
  --test-timesteps 5000 `
  --monitor none `
  --print-every-steps 50


"""

from __future__ import print_function

import argparse
from pathlib import Path

import torch
import pygame

from networks.on_policy.ppo.ppo_agent_v3 import PPOAgent
from simulation.carla_connection_v3 import ClientConnection
from simulation.carla_environment_rgb_v3 import CarlaEnvironmentRGB
from simulation.carla_sensors_v3 import CameraSensorEnv


DEFAULT_MODEL_NAME = "automav3_rgb_v3"


def parse_spawn_numbers(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                "Spawn numbers are 1-based and must be > 0."
            )
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("At least one spawn is required.")
    return list(dict.fromkeys(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one exact PPO V3 checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Exact PPO V3 .pth checkpoint path.",
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--town", type=str, default="current")
    parser.add_argument(
        "--safe-spawns",
        type=parse_spawn_numbers,
        default=parse_spawn_numbers("1,2,3,4"),
    )
    parser.add_argument(
        "--desired-speed",
        type=float,
        default=0.40,
        help="Evaluation target speed in real-equivalent m/s.",
    )
    parser.add_argument("--test-timesteps", type=int, default=5000)
    parser.add_argument("--max-episode-seconds", type=float, default=20.0)
    parser.add_argument(
        "--ppo-device", choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument(
        "--encoder-device", choices=["cpu", "cuda"], default="cpu"
    )
    parser.add_argument("--print-every-steps", type=int, default=50)
    parser.add_argument(
        "--monitor",
        choices=["none", "third", "ppo"],
        default="third",
        help=(
            "none=no window, third=rear third-person camera from "
            "carla_sensors_v3.CameraSensorEnv, "
            "ppo=show the exact front RGB frame used by VAE/PPO."
        ),
    )
    return parser.parse_args()


def inspect_checkpoint(path):
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint không phải dict PPO V3.")

    required = ("version", "obs_dim", "action_dim", "policy_state_dict")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError("Checkpoint thiếu field: {}".format(missing))

    if checkpoint["version"] != "PPO_V3_GD14":
        raise RuntimeError(
            "Sai version checkpoint: {}".format(checkpoint["version"])
        )
    if int(checkpoint["obs_dim"]) != 100:
        raise RuntimeError(
            "obs_dim={} != 100".format(checkpoint["obs_dim"])
        )
    if int(checkpoint["action_dim"]) != 2:
        raise RuntimeError(
            "action_dim={} != 2".format(checkpoint["action_dim"])
        )
    return checkpoint


class EvalStats(object):
    def __init__(self):
        self.total_steps = 0
        self.episodes = 0
        self.offroad = 0
        self.time_limit = 0
        self.other_done = 0
        self.total_reward = 0.0
        self.total_speed = 0.0
        self.total_abs_lat = 0.0
        self.total_abs_heading = 0.0
        self.episode_steps = 0
        self.episode_reward = 0.0

    def add_step(self, reward, info):
        imu = info.get("imu", {}) or {}
        self.total_steps += 1
        self.episode_steps += 1
        self.total_reward += float(reward)
        self.episode_reward += float(reward)
        self.total_speed += float(imu.get("speed_mps", 0.0))
        self.total_abs_lat += abs(float(info.get("lateral_error_m", 0.0)))
        self.total_abs_heading += abs(
            float(info.get("heading_error_rad", 0.0))
        )

    def finish_episode(self, reason):
        self.episodes += 1
        if reason == "offroad":
            self.offroad += 1
        elif reason == "time_limit":
            self.time_limit += 1
        else:
            self.other_done += 1

        print(
            "EPISODE {:04d} | steps={} | reward={:+.3f} | reason={}".format(
                self.episodes,
                self.episode_steps,
                self.episode_reward,
                reason,
            )
        )
        self.episode_steps = 0
        self.episode_reward = 0.0

    def print_final(self):
        n = max(1, self.total_steps)
        ep = max(1, self.episodes)
        print("")
        print("=" * 96)
        print("CHECKPOINT EVALUATION SUMMARY")
        print("=" * 96)
        print("steps              :", self.total_steps)
        print("episodes           :", self.episodes)
        print(
            "offroad            : {} ({:.1f}%)".format(
                self.offroad, 100.0 * self.offroad / float(ep)
            )
        )
        print(
            "time_limit         : {} ({:.1f}%)".format(
                self.time_limit, 100.0 * self.time_limit / float(ep)
            )
        )
        print("other_done         :", self.other_done)
        print(
            "mean_reward/step   : {:+.6f}".format(
                self.total_reward / float(n)
            )
        )
        print(
            "mean_speed_mps     : {:.4f}".format(
                self.total_speed / float(n)
            )
        )
        print(
            "mean_abs_lat_m     : {:.5f}".format(
                self.total_abs_lat / float(n)
            )
        )
        print(
            "mean_abs_heading   : {:.5f} rad".format(
                self.total_abs_heading / float(n)
            )
        )
        print("=" * 96)



def destroy_third_person_monitor(monitor):
    if monitor is None:
        return None

    sensor = getattr(monitor, "sensor", None)
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
    return None


class PPOFrontMonitor(object):
    """Show the exact RGB frame currently used by VAE/PPO."""

    def __init__(self, width=960, height=540):
        pygame.init()
        pygame.font.init()
        self.width = int(width)
        self.height = int(height)
        self.display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        pygame.display.set_caption(
            "PPO FRONT CAMERA - EXACT VAE/PPO INPUT"
        )
        self.font = pygame.font.SysFont("consolas", 22)
        self.small_font = pygame.font.SysFont("consolas", 18)
        self.enabled = True

    def pump_events(self):
        if not self.enabled:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.enabled = False

    def draw(self, env, action, reward, info):
        if not self.enabled:
            return
        self.pump_events()
        if not self.enabled:
            return

        camera_obj = getattr(env, "camera_obj", None)
        if camera_obj is None:
            return
        frames = getattr(camera_obj, "front_camera", None)
        if not frames:
            return

        rgb = frames[-1]
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        image_height = self.height - 120
        scaled = pygame.transform.smoothscale(
            surface, (self.width, image_height)
        )

        self.display.fill((0, 0, 0))
        self.display.blit(scaled, (0, 0))
        imu = info.get("imu", {}) or {}

        line1 = (
            "PPO steer={:+.3f}  speed_cmd={:.3f} m/s   "
            "speed={:.3f} m/s   reward={:+.4f}"
        ).format(
            float(action[0]),
            float(action[1]),
            float(imu.get("speed_mps", 0.0)),
            float(reward),
        )

        line2 = (
            "yaw={:+.3f} rad/s   ax={:+.3f} m/s2   "
            "lat={:+.4f} m   heading={:+.4f} rad"
        ).format(
            float(imu.get("yaw_rate_rad_s", 0.0)),
            float(imu.get("longitudinal_accel_mps2", 0.0)),
            float(info.get("lateral_error_m", 0.0)),
            float(info.get("heading_error_rad", 0.0)),
        )

        label1 = self.font.render(line1, True, (255, 255, 255))
        label2 = self.small_font.render(line2, True, (255, 255, 255))
        self.display.blit(label1, (12, image_height + 12))
        self.display.blit(label2, (12, image_height + 54))
        pygame.display.flip()

    def close(self):
        try:
            pygame.display.quit()
        except Exception:
            pass
        self.enabled = False


def create_monitor(mode, env):
    if mode == "third":
        print(
            "MONITOR: THIRD PERSON 720x720 "
            "(CameraSensorEnv from carla_sensors_v3.py)"
        )
        return CameraSensorEnv(env.vehicle), None

    if mode == "ppo":
        print(
            "MONITOR: PPO FRONT RGB "
            "(exact frame used by VAE/PPO)"
        )
        return None, PPOFrontMonitor()

    print("MONITOR: OFF")
    return None, None


def pump_third_person_events(monitor):
    if monitor is None:
        return
    try:
        pygame.event.pump()
    except Exception:
        pass


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(str(checkpoint_path))

    metadata = inspect_checkpoint(checkpoint_path)

    print("=" * 96)
    print("PPO V3 EXACT CHECKPOINT LOADER")
    print("=" * 96)
    print("checkpoint :", checkpoint_path)
    print("version    :", metadata["version"])
    print("obs_dim    :", metadata["obs_dim"])
    print("action_dim :", metadata["action_dim"])
    print("action_std :", metadata.get("action_std", "N/A"))
    print("mode       : deterministic EVAL (train=False)")
    print("DR         : OFF / nominal environment")
    print("=" * 96)

    env = None
    third_monitor = None
    ppo_monitor = None
    try:
        client, world = ClientConnection(args.town).setup()
        if client is None or world is None:
            raise RuntimeError("CARLA connection failed.")

        env = CarlaEnvironmentRGB(
            client=client,
            world=world,
            town=args.town,
            safe_spawn_numbers=args.safe_spawns,
            desired_speed_mps=float(args.desired_speed),
            max_episode_seconds=float(args.max_episode_seconds),
            encoder_device=args.encoder_device,
            domain_randomization_enabled=False,
            domain_randomization_seed=0,
        )

        agent = PPOAgent(
            town=args.model_name,
            action_std_init=float(metadata.get("action_std", 0.05)),
            device=args.ppo_device,
        )

        loaded_path = agent.load(checkpoint_path=str(checkpoint_path))
        print("")
        print("LOADED PPO :", loaded_path)

        obs = env.reset()

        third_monitor, ppo_monitor = create_monitor(
            args.monitor, env
        )

        if tuple(obs.shape) != (100,):
            raise RuntimeError("Observation shape sai: {}".format(obs.shape))

        stats = EvalStats()

        for step in range(1, int(args.test_timesteps) + 1):
            # train=False => deterministic policy action, no exploration noise.
            action = agent.get_action(obs, train=False)
            next_obs, reward, done, info = env.step(action)

            if third_monitor is not None:
                pump_third_person_events(third_monitor)

            if ppo_monitor is not None:
                ppo_monitor.draw(
                    env=env,
                    action=action,
                    reward=reward,
                    info=info,
                )

            stats.add_step(reward=reward, info=info)

            if (
                args.print_every_steps > 0
                and (step % args.print_every_steps == 0 or done)
            ):
                imu = info.get("imu", {}) or {}
                print(
                    "step={:06d} | action=({:+.3f},{:.3f}) | "
                    "speed={:.3f} | lat={:+.4f} | heading={:+.4f} | "
                    "reward={:+.4f} | done={} | reason={}".format(
                        step,
                        float(action[0]),
                        float(action[1]),
                        float(imu.get("speed_mps", 0.0)),
                        float(info.get("lateral_error_m", 0.0)),
                        float(info.get("heading_error_rad", 0.0)),
                        float(reward),
                        bool(done),
                        info.get("termination_reason"),
                    )
                )

            if done:
                stats.finish_episode(info.get("termination_reason"))

                if step < args.test_timesteps:
                    if third_monitor is not None:
                        third_monitor = destroy_third_person_monitor(
                            third_monitor
                        )

                    obs = env.reset()

                    if args.monitor == "third":
                        third_monitor = CameraSensorEnv(env.vehicle)
                else:
                    obs = next_obs
            else:
                obs = next_obs

        stats.print_final()
        print("")
        print("RESULT: EXACT CHECKPOINT LOAD/EVAL PASS")

    finally:
        if third_monitor is not None:
            third_monitor = destroy_third_person_monitor(
                third_monitor
            )

        if ppo_monitor is not None:
            try:
                ppo_monitor.close()
            except Exception:
                pass

        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
