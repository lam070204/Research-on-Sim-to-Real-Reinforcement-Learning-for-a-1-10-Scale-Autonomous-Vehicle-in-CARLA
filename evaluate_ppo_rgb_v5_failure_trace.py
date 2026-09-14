# -*- coding: utf-8 -*-
"""
PPO V5 FAILURE TRACE EVALUATOR

Deterministic CARLA evaluation for PPO V5 with failure diagnostics.
Designed for systematic auxiliary-map failures such as a repeated offroad
near the same curve/spawn.

Key properties
--------------
- No learning / no checkpoint overwrite.
- CLEAN by default: Dynamics/Vision/Sensor DR OFF, recovery scenarios OFF.
- Keeps a pre-failure ring buffer, then records every step until termination.
- Saves one CSV trace per suspicious/failed episode.
- Saves failure_index.csv and prints simple progress-position failure clusters.
- Logs raw PPO action, applied CARLA control, signed lateral/heading error,
  state-aware speed, vehicle world pose, weather, observation proprioception,
  latent norm/delta and optionally latent95 values.
- Stores all scalar/string/bool info entries as JSON for keys that may differ
  between environment revisions.

Python 3.7+
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, deque

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
            raise argparse.ArgumentTypeError("Spawn numbers are 1-based and must be > 0.")
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("At least one spawn is required.")
    return list(dict.fromkeys(values))


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate PPO V5 with per-step failure traces. CLEAN by default."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--carla-timeout", type=float, default=120.0)
    p.add_argument("--expected-map", default="maptrang/mapoval_white")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=9505)
    p.add_argument("--desired-speed", type=float, default=1.0)
    p.add_argument("--safe-spawns", type=parse_spawn_numbers, default=parse_spawn_numbers("1"))
    p.add_argument("--max-episode-seconds", type=float, default=60.0)

    # Diagnostic evaluator is CLEAN by default. Turn one source on explicitly.
    p.add_argument("--dynamics-dr", type=boolean_string, default=False)
    p.add_argument("--vision-dr", type=boolean_string, default=False)
    p.add_argument("--sensor-dr", type=boolean_string, default=False)
    p.add_argument("--recovery-scenarios", type=boolean_string, default=False)

    # Kept for controlled recovery experiments when explicitly enabled.
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
    p.add_argument("--csv", default=None, help="Episode summary CSV.")
    p.add_argument("--print-every-steps", type=int, default=0)

    # Failure trace options.
    p.add_argument("--trace", type=boolean_string, default=True)
    p.add_argument("--trace-dir", default="eval_results/failure_trace")
    p.add_argument("--trace-pre-seconds", type=float, default=4.0,
                   help="Keep this many seconds before the first diagnostic trigger.")
    p.add_argument("--trace-lateral-m", type=float, default=0.030,
                   help="Start preserving the trace when abs(lateral) reaches this value.")
    p.add_argument("--trace-heading-deg", type=float, default=8.0,
                   help="Start preserving the trace when abs(heading error) reaches this value.")
    p.add_argument("--trace-latent", type=boolean_string, default=True,
                   help="Store latent95 as JSON in each trace row.")
    p.add_argument("--trace-all-episodes", type=boolean_string, default=False,
                   help="Save trace even if no threshold/failure occurred.")
    p.add_argument("--cluster-gap-m", type=float, default=1.0,
                   help="Progress gap used for simple repeated-failure clustering.")
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
            "--desired-speed is locked to 1.0 so targets remain 0.60/0.50/0.40/0.30 m/s."
        )
    if a.trace_pre_seconds < 0.0:
        raise ValueError("--trace-pre-seconds must be >= 0")
    if a.trace_lateral_m < 0.0:
        raise ValueError("--trace-lateral-m must be >= 0")
    if a.trace_heading_deg < 0.0:
        raise ValueError("--trace-heading-deg must be >= 0")
    if a.cluster_gap_m <= 0.0:
        raise ValueError("--cluster-gap-m must be > 0")
    for name in ("recovery_spawn_prob", "disturbance_prob"):
        value = float(getattr(a, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError("{} must be in [0,1].".format(name))


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
        raise RuntimeError("Wrong checkpoint version: {!r}".format(ckpt.get("version")))
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
            "CARLA map mismatch: actual='{}', expected contains '{}'".format(map_name, a.expected_map)
        )
    # Fixed clean weather before env creation. Vision DR may override only when explicitly enabled.
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


def safe_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return float(default)


def finite_or_blank(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else ""
    except Exception:
        return ""


def find_vehicle(env):
    # Environment revisions may use different names. Never fail the evaluation
    # just because a diagnostic attribute is absent.
    for name in ("vehicle", "ego_vehicle", "car", "actor", "_vehicle"):
        obj = getattr(env, name, None)
        if obj is not None and hasattr(obj, "get_transform"):
            return obj
    for holder_name in ("vehicle_controller", "controller", "action_controller"):
        holder = getattr(env, holder_name, None)
        if holder is None:
            continue
        for name in ("vehicle", "ego_vehicle", "actor", "car"):
            obj = getattr(holder, name, None)
            if obj is not None and hasattr(obj, "get_transform"):
                return obj
    return None


def vehicle_snapshot(env):
    out = {
        "world_x": float("nan"),
        "world_y": float("nan"),
        "world_z": float("nan"),
        "vehicle_yaw_deg": float("nan"),
        "vehicle_pitch_deg": float("nan"),
        "vehicle_roll_deg": float("nan"),
        "applied_steer": float("nan"),
        "applied_throttle": float("nan"),
        "applied_brake": float("nan"),
    }
    vehicle = find_vehicle(env)
    if vehicle is None:
        return out
    try:
        tr = vehicle.get_transform()
        out.update({
            "world_x": float(tr.location.x),
            "world_y": float(tr.location.y),
            "world_z": float(tr.location.z),
            "vehicle_yaw_deg": float(tr.rotation.yaw),
            "vehicle_pitch_deg": float(tr.rotation.pitch),
            "vehicle_roll_deg": float(tr.rotation.roll),
        })
    except Exception:
        pass
    try:
        control = vehicle.get_control()
        out.update({
            "applied_steer": float(control.steer),
            "applied_throttle": float(control.throttle),
            "applied_brake": float(control.brake),
        })
    except Exception:
        pass
    return out


def weather_snapshot(world):
    names = (
        "cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
        "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
        "wetness", "fog_falloff", "scattering_intensity", "mie_scattering_scale",
        "rayleigh_scattering_scale", "dust_storm",
    )
    out = {}
    try:
        w = world.get_weather()
        for name in names:
            if hasattr(w, name):
                out["weather_{}".format(name)] = safe_float(getattr(w, name))
    except Exception:
        pass
    return out


def scalar_info_json(info):
    data = {}
    for key, value in info.items():
        if isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            data[str(key)] = value
        elif isinstance(value, np.generic):
            item = value.item()
            if isinstance(item, (bool, int, float, str)):
                data[str(key)] = item
    try:
        return json.dumps(data, sort_keys=True, separators=(",", ":"))
    except Exception:
        return "{}"


def first_info_float(info, keys, default=float("nan")):
    for key in keys:
        if key in info:
            try:
                return float(info[key])
            except Exception:
                pass
    return float(default)


def make_trace_record(env, world, episode_index, step, observation, action, reward, info,
                      prev_latent, save_latent):
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    latent = obs[:95] if obs.size >= 95 else np.asarray([], dtype=np.float32)
    latent_norm = float(np.linalg.norm(latent)) if latent.size else float("nan")
    latent_delta = float("nan")
    if prev_latent is not None and latent.size == prev_latent.size and latent.size > 0:
        latent_delta = float(np.linalg.norm(latent - prev_latent))

    row = {
        "episode": int(episode_index),
        "step": int(step),
        "approx_time_s": float(step) / 50.0,
        "reward": safe_float(reward, 0.0),
        "recovery_state": str(info.get("recovery_state", "unknown")),
        "adaptive_target_speed_mps": safe_float(info.get("adaptive_target_speed_mps", 0.0), 0.0),
        "speed_cmd_mps": safe_float(info.get("speed_cmd_mps", action[1]), action[1]),
        "speed_mps": safe_float(info.get("speed_mps", 0.0), 0.0),
        "ppo_steer_raw": safe_float(action[0], 0.0),
        "ppo_speed_raw": safe_float(action[1], 0.0),
        "lateral_error_m": safe_float(info.get("lateral_error_m", 0.0), 0.0),
        "heading_error_deg": math.degrees(safe_float(info.get("heading_error_rad", 0.0), 0.0)),
        "forward_progress_m": safe_float(info.get("forward_progress_m", 0.0), 0.0),
        "gyro_z": first_info_float(info, ("gyro_z", "gyroscope_z", "imu_gyro_z", "yaw_rate", "yaw_rate_rps")),
        "accel_x": first_info_float(info, ("accel_x", "accelerometer_x", "imu_accel_x")),
        "accel_y": first_info_float(info, ("accel_y", "accelerometer_y", "imu_accel_y")),
        "accel_z": first_info_float(info, ("accel_z", "accelerometer_z", "imu_accel_z")),
        "camera_lag_ticks": first_info_float(info, ("camera_lag_ticks", "camera_lag", "camera_frame_lag", "camera_lag_tick")),
        "offroad": bool(info.get("offroad", False)),
        "terminated": bool(info.get("terminated", False)),
        "truncated": bool(info.get("truncated", False)),
        "termination_reason": str(info.get("termination_reason", "")),
        "recovery_trigger_event": bool(info.get("recovery_trigger_event", False)),
        "recovery_success_event": bool(info.get("recovery_success_event", False)),
        "latent_l2": latent_norm,
        "latent_delta_l2": latent_delta,
        "obs_speed_feature": safe_float(obs[95]) if obs.size > 95 else float("nan"),
        "obs_yaw_feature": safe_float(obs[96]) if obs.size > 96 else float("nan"),
        "obs_accel_feature": safe_float(obs[97]) if obs.size > 97 else float("nan"),
        "obs_prev_steer": safe_float(obs[98]) if obs.size > 98 else float("nan"),
        "obs_prev_speed": safe_float(obs[99]) if obs.size > 99 else float("nan"),
        "latent95_json": json.dumps([float(x) for x in latent.tolist()], separators=(",", ":")) if save_latent else "",
        "info_json": scalar_info_json(info),
    }
    row.update(vehicle_snapshot(env))
    row.update(weather_snapshot(world))
    return row, latent.copy() if latent.size else None


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


def is_failure_reason(reason):
    text = str(reason).strip().lower()
    good = ("time_limit", "canary_horizon", "horizon")
    if any(x in text for x in good):
        return False
    bad = ("offroad", "collision", "stuck", "lane", "failure", "crash")
    return any(x in text for x in bad) or bool(text)


def print_episode(result):
    attempts = int(result["recovery_attempts"])
    successes = int(result["recovery_successes"])
    recovery_text = "{}/{}".format(successes, attempts)
    if attempts > 0:
        recovery_text += " ({:.1f}%)".format(100.0 * successes / float(attempts))
    print(
        "EP {:03d} | reason={:12s} | steps={:4d} | reward={:+8.3f} | "
        "progress={:7.2f}m | v_avg={:.3f} v_max={:.3f} | "
        "|lat|avg={:.3f} max={:.3f}m | |head|avg={:.1f} max={:.1f}deg | recovery={}".format(
            int(result["episode"]), str(result["reason"])[:12], int(result["steps"]),
            float(result["reward"]), float(result["progress_m"]),
            float(result["avg_speed_mps"]), float(result["max_speed_mps"]),
            float(result["avg_abs_lateral_m"]), float(result["max_abs_lateral_m"]),
            float(result["avg_abs_heading_deg"]), float(result["max_abs_heading_deg"]),
            recovery_text,
        )
    )


def print_summary(results):
    print("\n" + "=" * 118)
    print("V5 FAILURE TRACE EVALUATION SUMMARY")
    print("=" * 118)
    reasons = Counter(str(item["reason"]) for item in results)
    total_attempts = sum(int(x["recovery_attempts"]) for x in results)
    total_successes = sum(int(x["recovery_successes"]) for x in results)
    print("episodes             :", len(results))
    print("termination reasons  :", dict(reasons))
    print("reward avg           : {:+.4f}".format(mean([x["reward"] for x in results])))
    print("progress avg         : {:.3f} m".format(mean([x["progress_m"] for x in results])))
    print("speed avg            : {:.3f} m/s".format(mean([x["avg_speed_mps"] for x in results])))
    print("|lateral| avg        : {:.4f} m".format(mean([x["avg_abs_lateral_m"] for x in results])))
    print("|lateral| max avg    : {:.4f} m".format(mean([x["max_abs_lateral_m"] for x in results])))
    print("|heading| avg        : {:.3f} deg".format(mean([x["avg_abs_heading_deg"] for x in results])))
    print("recovery success     : {}/{} ({:.1f}%)".format(
        total_successes, total_attempts,
        100.0 * total_successes / float(total_attempts) if total_attempts > 0 else 0.0,
    ))
    print("\nSTATE-SPEED:")
    for state_name in STATE_NAMES:
        count = sum(int(x["{}_count".format(state_name)]) for x in results)
        if count <= 0:
            print("  {:8s} | n=0".format(state_name.upper()))
            continue
        sums = {"target": 0.0, "cmd": 0.0, "speed": 0.0, "cmd_abs_error": 0.0, "speed_abs_error": 0.0}
        for item in results:
            n = int(item["{}_count".format(state_name)])
            for key in sums:
                sums[key] += float(item["{}_{}".format(state_name, key)]) * n
        print(
            "  {:8s} | n={:6d} | target={:.3f} | cmd={:.3f} | speed={:.3f} | "
            "|cmd-target|={:.3f} | |v-target|={:.3f}".format(
                state_name.upper(), count,
                sums["target"] / count, sums["cmd"] / count, sums["speed"] / count,
                sums["cmd_abs_error"] / count, sums["speed_abs_error"] / count,
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
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(item)
    print("CSV:", path)


def write_trace_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float) and not math.isfinite(value):
                    value = ""
                cleaned[key] = value
            writer.writerow(cleaned)


def print_failure_clusters(failure_rows, gap_m):
    usable = [x for x in failure_rows if isinstance(x.get("progress_m"), (int, float))]
    usable = [x for x in usable if math.isfinite(float(x["progress_m"]))]
    if not usable:
        print("FAILURE CLUSTERS: no failed episodes with progress data.")
        return
    usable.sort(key=lambda x: float(x["progress_m"]))
    clusters = []
    current = [usable[0]]
    for item in usable[1:]:
        if float(item["progress_m"]) - float(current[-1]["progress_m"]) > float(gap_m):
            clusters.append(current)
            current = [item]
        else:
            current.append(item)
    clusters.append(current)

    print("\n" + "=" * 118)
    print("FAILURE CLUSTERS | simple clustering by terminal progress gap > {:.2f} m".format(gap_m))
    print("=" * 118)
    for idx, cluster in enumerate(clusters, 1):
        progs = np.asarray([float(x["progress_m"]) for x in cluster], dtype=np.float64)
        steps = np.asarray([float(x["fail_step"]) for x in cluster], dtype=np.float64)
        xs = np.asarray([float(x["world_x"]) for x in cluster if x.get("world_x") not in ("", None) and math.isfinite(float(x["world_x"]))], dtype=np.float64)
        ys = np.asarray([float(x["world_y"]) for x in cluster if x.get("world_y") not in ("", None) and math.isfinite(float(x["world_y"]))], dtype=np.float64)
        pose_text = ""
        if xs.size and ys.size:
            pose_text = " | pos=({:.2f}±{:.2f},{:.2f}±{:.2f})".format(
                float(xs.mean()), float(xs.std()), float(ys.mean()), float(ys.std())
            )
        print(
            "  cluster {:02d} | n={:3d} | progress={:.2f}±{:.2f}m [{:.2f},{:.2f}] | "
            "step={:.1f}±{:.1f}{}".format(
                idx, len(cluster), float(progs.mean()), float(progs.std()),
                float(progs.min()), float(progs.max()),
                float(steps.mean()), float(steps.std()), pose_text,
            )
        )
    print("=" * 118)


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
    trace_root = os.path.abspath(os.path.expanduser(a.trace_dir))
    if a.trace:
        os.makedirs(trace_root, exist_ok=True)

    try:
        env = CarlaEnvironmentRGBV5(
            client=client,
            world=world,
            town="__v5_failure_trace_eval__",
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
            town="__v5_failure_trace_eval__",
            action_std_init=float(meta.get("action_std", 0.05)),
            device=a.device,
        )
        loaded = agent.load(checkpoint_path=checkpoint)
        agent.memory.clear()
        _ = env.reset()
        profile = verify_reward_profile(env)

        print("=" * 118)
        print("PPO V5 FAILURE TRACE - DETERMINISTIC EVALUATION")
        print("=" * 118)
        print("checkpoint        :", loaded)
        print("map               :", map_name)
        print("episodes          :", a.episodes)
        print("safe spawns       :", list(a.safe_spawns))
        print("DR                : dynamics={} | vision={} | sensor={}".format(
            a.dynamics_dr, a.vision_dr, a.sensor_dr))
        print("recovery scenarios:", a.recovery_scenarios)
        print("speed profile     : stable={:.2f} mild={:.2f} moderate={:.2f} strong={:.2f}".format(
            profile["stable"], profile["mild"], profile["moderate"], profile["strong"]))
        print("trace             : {} | pre={:.1f}s | lat>={:.3f}m or head>={:.1f}deg | latent={}".format(
            a.trace, a.trace_pre_seconds, a.trace_lateral_m, a.trace_heading_deg, a.trace_latent))
        if a.trace:
            print("trace dir         :", trace_root)
        print("=" * 118)

        results = []
        failure_rows = []
        pre_steps = max(1, int(round(float(a.trace_pre_seconds) * 50.0)))

        for episode_index in range(1, int(a.episodes) + 1):
            observation = env.reset()
            metrics = EpisodeMetrics()
            episode_start = time.time()
            step = 0
            prebuffer = deque(maxlen=pre_steps)
            captured_rows = None
            first_trigger_step = None
            first_trigger_reason = ""
            prev_latent = None
            final_record = None

            while True:
                step += 1
                with torch.no_grad():
                    action = agent.get_action(observation, train=False)

                next_observation, reward, done, info = env.step(action)
                metrics.add(action=action, reward=reward, info=info)

                record, current_latent = make_trace_record(
                    env=env,
                    world=world,
                    episode_index=episode_index,
                    step=step,
                    observation=observation,
                    action=action,
                    reward=reward,
                    info=info,
                    prev_latent=prev_latent,
                    save_latent=bool(a.trace_latent),
                )
                prev_latent = current_latent
                final_record = record

                abs_lat = abs(float(record["lateral_error_m"]))
                abs_head = abs(float(record["heading_error_deg"]))
                threshold_hit = (
                    abs_lat >= float(a.trace_lateral_m)
                    or abs_head >= float(a.trace_heading_deg)
                )

                if a.trace:
                    if captured_rows is None:
                        prebuffer.append(record)
                        if threshold_hit:
                            captured_rows = list(prebuffer)
                            first_trigger_step = int(step)
                            if abs_lat >= float(a.trace_lateral_m) and abs_head >= float(a.trace_heading_deg):
                                first_trigger_reason = "lateral+heading"
                            elif abs_lat >= float(a.trace_lateral_m):
                                first_trigger_reason = "lateral"
                            else:
                                first_trigger_reason = "heading"
                            print(
                                "TRACE TRIGGER | ep={} step={} progress={:.3f}m | lat={:+.4f}m head={:+.2f}deg | "
                                "steer={:+.3f} speed={:.3f} cmd={:.3f} | reason={}".format(
                                    episode_index, step, float(record["forward_progress_m"]),
                                    float(record["lateral_error_m"]), float(record["heading_error_deg"]),
                                    float(record["ppo_steer_raw"]), float(record["speed_mps"]),
                                    float(record["speed_cmd_mps"]), first_trigger_reason,
                                )
                            )
                    else:
                        captured_rows.append(record)

                if a.print_every_steps > 0 and step % int(a.print_every_steps) == 0:
                    print(
                        "  ep={:03d} step={:04d} | state={:8s} target={:.3f} cmd={:.3f} speed={:.3f} | "
                        "lat={:+.3f}m head={:+.1f}deg | steer={:+.3f} applied={:+.3f} | progress={:.2f}".format(
                            episode_index, step, str(record["recovery_state"]).upper(),
                            float(record["adaptive_target_speed_mps"]), float(record["speed_cmd_mps"]),
                            float(record["speed_mps"]), float(record["lateral_error_m"]),
                            float(record["heading_error_deg"]), float(record["ppo_steer_raw"]),
                            safe_float(record.get("applied_steer")), float(record["forward_progress_m"]),
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

                    bad_reason = is_failure_reason(result["reason"])
                    should_save_trace = bool(a.trace) and (
                        bool(a.trace_all_episodes) or captured_rows is not None or bad_reason
                    )
                    if should_save_trace:
                        if captured_rows is None:
                            captured_rows = list(prebuffer)
                            if final_record is not None and (not captured_rows or captured_rows[-1] is not final_record):
                                captured_rows.append(final_record)
                        trace_path = os.path.join(
                            trace_root,
                            "episode_{:03d}_{}_trace.csv".format(
                                episode_index,
                                str(result["reason"] or "done").replace(" ", "_").replace("/", "_"),
                            ),
                        )
                        write_trace_csv(trace_path, captured_rows)
                        print("  TRACE CSV:", trace_path)

                    if bad_reason:
                        snap = final_record or {}
                        failure_rows.append({
                            "episode": int(episode_index),
                            "reason": str(result["reason"]),
                            "first_trigger_step": int(first_trigger_step) if first_trigger_step is not None else "",
                            "first_trigger_reason": first_trigger_reason,
                            "fail_step": int(result["steps"]),
                            "progress_m": float(result["progress_m"]),
                            "world_x": finite_or_blank(snap.get("world_x", float("nan"))),
                            "world_y": finite_or_blank(snap.get("world_y", float("nan"))),
                            "vehicle_yaw_deg": finite_or_blank(snap.get("vehicle_yaw_deg", float("nan"))),
                            "final_lateral_error_m": finite_or_blank(snap.get("lateral_error_m", float("nan"))),
                            "final_heading_error_deg": finite_or_blank(snap.get("heading_error_deg", float("nan"))),
                            "final_ppo_steer": finite_or_blank(snap.get("ppo_steer_raw", float("nan"))),
                            "final_applied_steer": finite_or_blank(snap.get("applied_steer", float("nan"))),
                            "final_speed_mps": finite_or_blank(snap.get("speed_mps", float("nan"))),
                            "final_speed_cmd_mps": finite_or_blank(snap.get("speed_cmd_mps", float("nan"))),
                            "camera_lag_ticks": finite_or_blank(snap.get("camera_lag_ticks", float("nan"))),
                            "max_abs_lateral_m": float(result["max_abs_lateral_m"]),
                            "max_abs_heading_deg": float(result["max_abs_heading_deg"]),
                            "avg_abs_steer": float(result["avg_abs_steer"]),
                        })
                    break

        print_summary(results)
        save_csv(a.csv, results)

        if a.trace:
            failure_index_path = os.path.join(trace_root, "failure_index.csv")
            if failure_rows:
                write_trace_csv(failure_index_path, failure_rows)
                print("FAILURE INDEX:", failure_index_path)
            print_failure_clusters(failure_rows, a.cluster_gap_m)

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        print("V5 failure-trace evaluation cleanup complete.")


if __name__ == "__main__":
    main()
