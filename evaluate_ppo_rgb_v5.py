# -*- coding: utf-8 -*-
"""
PPO V5 CLEAN - CHECKPOINT EVALUATOR

Evaluate a trained PPO V5 checkpoint in CARLA.
- V5 modules only.
- Deterministic policy: train=False.
- No learning, no checkpoint overwrite, no training-state mutation.
- Per-episode and aggregate diagnostics.
- Optional CSV output.

Python 3.7+
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from networks.on_policy.ppo.ppo_agent_v5 import PPOAgent
from simulation.carla_connection_v5 import carla
from simulation.carla_environment_rgb_v5 import (
    CarlaEnvironmentRGBV5,
    configure_recovery_scenarios_v5,
)

STATE_NAMES = ("stable", "mild", "moderate", "strong")
EXPECTED_SPEED_PROFILE = {
    "stable": 0.60,
    "mild": 0.50,
    "moderate": 0.40,
    "strong": 0.30,
}


def boolean_string(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Use true/false.")


def parse_spawn_numbers(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number <= 0:
            raise argparse.ArgumentTypeError(
                "Spawn numbers are 1-based and must be > 0."
            )
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("At least one spawn is required.")
    return list(dict.fromkeys(values))


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a PPO V5 CLEAN checkpoint in CARLA."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-timeout", type=float, default=120.0)
    p.add_argument("--expected-map", default="maptrang/mapoval_white")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=9505)
    p.add_argument("--desired-speed", type=float, default=1.0)
    p.add_argument(
        "--safe-spawns",
        type=parse_spawn_numbers,
        default=parse_spawn_numbers("1,2,3,4"),
    )
    p.add_argument("--max-episode-seconds", type=float, default=60.0)

    # Match current V5 training distribution by default.
    p.add_argument("--dynamics-dr", type=boolean_string, default=True)
    p.add_argument("--vision-dr", type=boolean_string, default=True)
    p.add_argument("--sensor-dr", type=boolean_string, default=True)
    p.add_argument("--recovery-scenarios", type=boolean_string, default=True)

    p.add_argument("--recovery-spawn-prob", type=float, default=0.75)
    p.add_argument("--offset-min-m", type=float, default=0.055)
    p.add_argument("--offset-max-m", type=float, default=0.105)
    p.add_argument("--heading-min-deg", type=float, default=5.0)
    p.add_argument("--heading-max-deg", type=float, default=12.0)

    p.add_argument("--disturbance-prob", type=float, default=0.65)
    p.add_argument("--disturbance-start-min-s", type=float, default=1.2)
    p.add_argument("--disturbance-start-max-s", type=float, default=3.2)
    p.add_argument("--disturbance-duration-min-s", type=float, default=0.10)
    p.add_argument("--disturbance-duration-max-s", type=float, default=0.22)
    p.add_argument("--disturbance-steer-min", type=float, default=0.14)
    p.add_argument("--disturbance-steer-max", type=float, default=0.28)

    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--csv", default=None)
    p.add_argument(
        "--print-every-steps",
        type=int,
        default=0,
        help="0 disables per-step diagnostics.",
    )
    return p.parse_args()


def validate_args(a):
    if a.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if a.max_episode_seconds <= 0.0:
        raise ValueError("--max-episode-seconds must be > 0")
    if a.torch_threads <= 0:
        raise ValueError("--torch-threads must be > 0")
    if abs(float(a.desired_speed) - 1.0) > 1e-9:
        raise ValueError(
            "--desired-speed is locked to 1.0 so targets remain "
            "0.60/0.55/0.45/0.35 m/s."
        )
    for name in ("recovery_spawn_prob", "disturbance_prob"):
        value = float(getattr(a, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError("{} must be in [0,1].".format(name))
    if not (0.0 <= a.offset_min_m <= a.offset_max_m):
        raise ValueError("Invalid recovery offset range.")
    if not (0.0 <= a.heading_min_deg <= a.heading_max_deg):
        raise ValueError("Invalid recovery heading range.")
    if not (
        0.0 < a.disturbance_duration_min_s <= a.disturbance_duration_max_s
    ):
        raise ValueError("Invalid disturbance duration range.")
    if not (
        0.0 <= a.disturbance_steer_min <= a.disturbance_steer_max <= 1.0
    ):
        raise ValueError("Invalid disturbance steer range.")


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inspect_checkpoint(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint is not a dict.")
    if ckpt.get("version") != "PPO_V5_CLEAN":
        raise RuntimeError(
            "Wrong checkpoint version: {!r}; expected 'PPO_V5_CLEAN'.".format(
                ckpt.get("version")
            )
        )
    if int(ckpt.get("obs_dim", -1)) != 100:
        raise RuntimeError("Checkpoint obs_dim != 100.")
    if int(ckpt.get("action_dim", -1)) != 2:
        raise RuntimeError("Checkpoint action_dim != 2.")
    if "policy_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing policy_state_dict.")
    return ckpt


def connect_carla(a):
    client = carla.Client(str(a.host), int(a.port))
    client.set_timeout(float(a.carla_timeout))
    world = client.get_world()
    map_name = str(world.get_map().name)

    expected = str(a.expected_map).replace("\\", "/").lower()
    actual = map_name.replace("\\", "/").lower()
    if expected and expected not in actual:
        raise RuntimeError(
            "CARLA map mismatch: actual='{}', expected contains '{}'".format(
                map_name, a.expected_map
            )
        )

    world.set_weather(carla.WeatherParameters.CloudyNoon)
    return client, world, map_name


def verify_reward_profile(env):
    reward_fn = env.reward_manager.reward_fn
    profile = {}
    for state_name in STATE_NAMES:
        profile[state_name] = float(
            reward_fn.target_speed_for_state(
                cruise_target_speed_mps=1.0,
                recovery_state=state_name,
            )
        )
    for state_name, expected in EXPECTED_SPEED_PROFILE.items():
        actual = float(profile[state_name])
        if abs(actual - float(expected)) > 1e-6:
            raise RuntimeError(
                "REWARD PROFILE MISMATCH | {} actual={} expected={}".format(
                    state_name, actual, expected
                )
            )
    return profile


class EpisodeMetrics(object):
    def __init__(self):
        self.reward = 0.0
        self.steps = 0
        self.progress_m = 0.0
        self.speed_sum = 0.0
        self.speed_max = 0.0
        self.cmd_speed_sum = 0.0
        self.abs_steer_sum = 0.0
        self.abs_lat_sum = 0.0
        self.abs_lat_max = 0.0
        self.abs_heading_sum = 0.0
        self.abs_heading_max = 0.0
        self.recovery_attempts = 0
        self.recovery_successes = 0
        self.state = {}
        for name in STATE_NAMES:
            self.state[name] = {
                "count": 0,
                "target_sum": 0.0,
                "cmd_sum": 0.0,
                "speed_sum": 0.0,
                "abs_cmd_error_sum": 0.0,
                "abs_speed_error_sum": 0.0,
            }

    def add(self, action, reward, info):
        self.reward += float(reward)
        self.steps += 1
        speed = float(info.get("speed_mps", 0.0))
        cmd = float(info.get("speed_cmd_mps", float(action[1])))
        lat = abs(float(info.get("lateral_error_m", 0.0)))
        heading = abs(float(info.get("heading_error_rad", 0.0)))

        self.speed_sum += speed
        self.speed_max = max(self.speed_max, speed)
        self.cmd_speed_sum += cmd
        self.abs_steer_sum += abs(float(action[0]))
        self.abs_lat_sum += lat
        self.abs_lat_max = max(self.abs_lat_max, lat)
        self.abs_heading_sum += heading
        self.abs_heading_max = max(self.abs_heading_max, heading)
        self.progress_m += float(info.get("forward_progress_m", 0.0))

        if bool(info.get("recovery_trigger_event", False)):
            self.recovery_attempts += 1
        if bool(info.get("recovery_success_event", False)):
            self.recovery_successes += 1

        state_name = str(info.get("recovery_state", "unknown"))
        if state_name in self.state:
            target = float(info.get("adaptive_target_speed_mps", 0.0))
            values = self.state[state_name]
            values["count"] += 1
            values["target_sum"] += target
            values["cmd_sum"] += cmd
            values["speed_sum"] += speed
            values["abs_cmd_error_sum"] += abs(cmd - target)
            values["abs_speed_error_sum"] += abs(speed - target)

    def finish(self, info, episode_index, wall_s):
        n = max(self.steps, 1)
        result = {
            "episode": int(episode_index),
            "reward": float(self.reward),
            "steps": int(self.steps),
            "wall_s": float(wall_s),
            "progress_m": float(self.progress_m),
            "avg_speed_mps": self.speed_sum / n,
            "max_speed_mps": float(self.speed_max),
            "avg_cmd_speed_mps": self.cmd_speed_sum / n,
            "avg_abs_steer": self.abs_steer_sum / n,
            "avg_abs_lateral_m": self.abs_lat_sum / n,
            "max_abs_lateral_m": float(self.abs_lat_max),
            "avg_abs_heading_deg": math.degrees(self.abs_heading_sum / n),
            "max_abs_heading_deg": math.degrees(self.abs_heading_max),
            "recovery_attempts": int(self.recovery_attempts),
            "recovery_successes": int(self.recovery_successes),
            "reason": str(info.get("termination_reason", "")),
            "terminated": bool(info.get("terminated", False)),
            "truncated": bool(info.get("truncated", False)),
        }

        for state_name in STATE_NAMES:
            values = self.state[state_name]
            count = int(values["count"])
            result["{}_count".format(state_name)] = count
            for suffix, key in (
                ("target", "target_sum"),
                ("cmd", "cmd_sum"),
                ("speed", "speed_sum"),
                ("cmd_abs_error", "abs_cmd_error_sum"),
                ("speed_abs_error", "abs_speed_error_sum"),
            ):
                result["{}_{}".format(state_name, suffix)] = (
                    float(values[key]) / float(count) if count > 0 else 0.0
                )
        return result


def mean(values):
    return float(sum(values)) / float(len(values)) if values else 0.0


def print_episode(result):
    attempts = int(result["recovery_attempts"])
    successes = int(result["recovery_successes"])
    recovery_text = "{}/{}".format(successes, attempts)
    if attempts > 0:
        recovery_text += " ({:.1f}%)".format(100.0 * successes / float(attempts))

    print(
        "EP {:03d} | reason={:10s} | steps={:4d} | reward={:+8.3f} | "
        "progress={:7.2f}m | v_avg={:.3f} v_max={:.3f} | "
        "|lat|avg={:.3f} max={:.3f}m | |head|avg={:.1f} max={:.1f}deg | "
        "recovery={}".format(
            int(result["episode"]),
            str(result["reason"])[:10],
            int(result["steps"]),
            float(result["reward"]),
            float(result["progress_m"]),
            float(result["avg_speed_mps"]),
            float(result["max_speed_mps"]),
            float(result["avg_abs_lateral_m"]),
            float(result["max_abs_lateral_m"]),
            float(result["avg_abs_heading_deg"]),
            float(result["max_abs_heading_deg"]),
            recovery_text,
        )
    )


def print_summary(results):
    print("\n" + "=" * 118)
    print("V5 POLICY EVALUATION SUMMARY")
    print("=" * 118)

    reasons = Counter(str(item["reason"]) for item in results)
    total_attempts = sum(int(x["recovery_attempts"]) for x in results)
    total_successes = sum(int(x["recovery_successes"]) for x in results)

    print("episodes             :", len(results))
    print("termination reasons  :", dict(reasons))
    print("reward avg           : {:+.4f}".format(mean([x["reward"] for x in results])))
    print("progress avg         : {:.3f} m".format(mean([x["progress_m"] for x in results])))
    print("speed avg            : {:.3f} m/s".format(mean([x["avg_speed_mps"] for x in results])))
    print("speed max avg        : {:.3f} m/s".format(mean([x["max_speed_mps"] for x in results])))
    print("|lateral| avg        : {:.4f} m".format(mean([x["avg_abs_lateral_m"] for x in results])))
    print("|lateral| max avg    : {:.4f} m".format(mean([x["max_abs_lateral_m"] for x in results])))
    print("|heading| avg        : {:.3f} deg".format(mean([x["avg_abs_heading_deg"] for x in results])))
    print(
        "recovery success     : {}/{} ({:.1f}%)".format(
            total_successes,
            total_attempts,
            100.0 * total_successes / float(total_attempts) if total_attempts > 0 else 0.0,
        )
    )

    print("\nSTATE-SPEED:")
    for state_name in STATE_NAMES:
        count = sum(int(x["{}_count".format(state_name)]) for x in results)
        if count <= 0:
            print("  {:8s} | n=0".format(state_name.upper()))
            continue

        sums = {
            "target": 0.0,
            "cmd": 0.0,
            "speed": 0.0,
            "cmd_abs_error": 0.0,
            "speed_abs_error": 0.0,
        }
        for item in results:
            n = int(item["{}_count".format(state_name)])
            for key in sums:
                sums[key] += float(item["{}_{}".format(state_name, key)]) * n

        print(
            "  {:8s} | n={:6d} | target={:.3f} | cmd={:.3f} | speed={:.3f} | "
            "|cmd-target|={:.3f} | |v-target|={:.3f}".format(
                state_name.upper(),
                count,
                sums["target"] / count,
                sums["cmd"] / count,
                sums["speed"] / count,
                sums["cmd_abs_error"] / count,
                sums["speed_abs_error"] / count,
            )
        )
    print("=" * 118)


def save_csv(path, results):
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = [
        "episode", "reason", "terminated", "truncated", "steps", "wall_s",
        "reward", "progress_m", "avg_speed_mps", "max_speed_mps",
        "avg_cmd_speed_mps", "avg_abs_steer", "avg_abs_lateral_m",
        "max_abs_lateral_m", "avg_abs_heading_deg", "max_abs_heading_deg",
        "recovery_attempts", "recovery_successes",
    ]
    for state_name in STATE_NAMES:
        fieldnames.extend([
            "{}_count".format(state_name),
            "{}_target".format(state_name),
            "{}_cmd".format(state_name),
            "{}_speed".format(state_name),
            "{}_cmd_abs_error".format(state_name),
            "{}_speed_abs_error".format(state_name),
        ])

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item)
    print("CSV:", path)


def main():
    a = parse_args()
    validate_args(a)
    torch.set_num_threads(int(a.torch_threads))
    seed_everything(int(a.seed))

    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for --device cuda.")
    if a.encoder_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for --encoder-device cuda.")

    checkpoint = os.path.abspath(os.path.expanduser(a.checkpoint))
    meta = inspect_checkpoint(checkpoint)

    configure_recovery_scenarios_v5(
        recovery_spawn_probability=float(a.recovery_spawn_prob),
        spawn_lateral_min_m=float(a.offset_min_m),
        spawn_lateral_max_m=float(a.offset_max_m),
        spawn_heading_min_deg=float(a.heading_min_deg),
        spawn_heading_max_deg=float(a.heading_max_deg),
        disturbance_probability=float(a.disturbance_prob),
        disturbance_start_min_s=float(a.disturbance_start_min_s),
        disturbance_start_max_s=float(a.disturbance_start_max_s),
        disturbance_duration_min_s=float(a.disturbance_duration_min_s),
        disturbance_duration_max_s=float(a.disturbance_duration_max_s),
        disturbance_steer_min=float(a.disturbance_steer_min),
        disturbance_steer_max=float(a.disturbance_steer_max),
        seed=int(a.seed),
    )

    client, world, map_name = connect_carla(a)
    env = None

    try:
        env = CarlaEnvironmentRGBV5(
            client=client,
            world=world,
            town="__v5_clean_eval__",
            safe_spawn_numbers=list(a.safe_spawns),
            desired_speed_mps=float(a.desired_speed),
            max_episode_seconds=float(a.max_episode_seconds),
            encoder_device=a.encoder_device,
            dynamics_dr_enabled=bool(a.dynamics_dr),
            vision_dr_enabled=bool(a.vision_dr),
            sensor_dr_enabled=bool(a.sensor_dr),
            domain_randomization_seed=int(a.seed) + 60000,
            recovery_scenarios_enabled=bool(a.recovery_scenarios),
        )

        agent = PPOAgent(
            town="__v5_clean_eval__",
            action_std_init=float(meta.get("action_std", 0.05)),
            device=a.device,
        )
        loaded = agent.load(checkpoint_path=checkpoint)
        agent.memory.clear()
        # Initialize env before reading reward_manager
        _ = env.reset()

        profile = verify_reward_profile(env)

        print("=" * 118)
        print("PPO V5 CLEAN - DETERMINISTIC EVALUATION")
        print("=" * 118)
        print("checkpoint        :", loaded)
        print("checkpoint version:", meta.get("version"))
        print("map               :", map_name)
        print("episodes          :", a.episodes)
        print("device            :", a.device)
        print("encoder device    :", a.encoder_device)
        print(
            "DR                : dynamics={} | vision={} | sensor={}".format(
                a.dynamics_dr, a.vision_dr, a.sensor_dr
            )
        )
        print(
            "recovery          : enabled={} | spawn={:.0f}% | disturbance={:.0f}%".format(
                a.recovery_scenarios,
                100.0 * a.recovery_spawn_prob,
                100.0 * a.disturbance_prob,
            )
        )
        print(
            "speed profile     : stable={:.2f} mild={:.2f} moderate={:.2f} strong={:.2f}".format(
                profile["stable"],
                profile["mild"],
                profile["moderate"],
                profile["strong"],
            )
        )
        print("policy mode       : deterministic | train=False")
        print("=" * 118)

        results = []
        for episode_index in range(1, int(a.episodes) + 1):
            observation = env.reset()
            metrics = EpisodeMetrics()
            episode_start = time.time()
            step = 0

            while True:
                step += 1
                with torch.no_grad():
                    action = agent.get_action(observation, train=False)

                next_observation, reward, done, info = env.step(action)
                metrics.add(action=action, reward=reward, info=info)

                if a.print_every_steps > 0 and step % int(a.print_every_steps) == 0:
                    print(
                        "  ep={:03d} step={:04d} | state={:8s} target={:.3f} "
                        "cmd={:.3f} speed={:.3f} | lat={:+.3f}m head={:+.1f}deg | "
                        "steer={:+.3f} reward={:+.4f}".format(
                            episode_index,
                            step,
                            str(info.get("recovery_state", "unknown")).upper(),
                            float(info.get("adaptive_target_speed_mps", 0.0)),
                            float(info.get("speed_cmd_mps", action[1])),
                            float(info.get("speed_mps", 0.0)),
                            float(info.get("lateral_error_m", 0.0)),
                            math.degrees(float(info.get("heading_error_rad", 0.0))),
                            float(action[0]),
                            float(reward),
                        )
                    )

                observation = next_observation
                if done:
                    result = metrics.finish(
                        info=info,
                        episode_index=episode_index,
                        wall_s=time.time() - episode_start,
                    )
                    results.append(result)
                    print_episode(result)
                    break

        print_summary(results)
        save_csv(a.csv, results)

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        print("V5 evaluation cleanup complete.")


if __name__ == "__main__":
    main()

