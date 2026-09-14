# V4_VISION_DR_TRAINER_COPY
# Source copied from: train_ppo_rgb_v3_verbose.py
# -*- coding: utf-8 -*-
"""
GĐ15 - Clean PPO V3 trainer for CARLA -> real 1/10 vehicle.

Canonical pipeline:
    env.reset() -> obs100
    PPO -> [steer_cmd, speed_cmd_mps]
    env.step()
    -> reward + terminated/truncated
    -> PPO rollout
    -> update every rollout_steps

Important:
- NO EncodeState/extra VAE call here. The environment already returns obs100.
- NO throttle action.
- NO privileged lateral/heading in policy observation.
- Time-limit truncation bootstraps V(next_obs).
- Mid-rollout updates bootstrap V(last_obs).

python .\train_ppo_rgb_v3_verbose.py `
  --train true `
  --load-checkpoint false `
  --print-every-steps 50
  
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from networks.on_policy.ppo.ppo_agent_v3 import PPOAgent
from simulation.carla_connection_v3 import ClientConnection
from simulation.carla_environment_rgb_v4_vision_dr_rightturn_safe import CarlaEnvironmentRGB

from parameters_v3 import (
    ACTION_STD_INIT,
    MODEL_LOAD,
    PPO_ACTION_STD_DECAY,
    PPO_ACTION_STD_DECAY_FREQ,
    PPO_ACTION_STD_MIN,
    PPO_CHECKPOINT_EVERY_STEPS,
    PPO_CURRICULUM,
    PPO_MAX_EPISODE_SECONDS,
    PPO_ROLLOUT_STEPS,
    SEED,
    TEST_TIMESTEPS,
    TOTAL_TIMESTEPS,
)


DEFAULT_MODEL_NAME = "automav3_rightturn35_nominal_v1"
DEFAULT_SAFE_SPAWNS = "1,2,3,4"


def boolean_string(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y", "on"}:
        return True

    if value in {"false", "0", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        "Invalid boolean value: {}. Use true/false.".format(value)
    )


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
        raise argparse.ArgumentTypeError(
            "At least one safe spawn is required."
        )

    return list(dict.fromkeys(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean PPO V3 trainer for AUTOMAV3."
    )

    parser.add_argument(
        "--train",
        type=boolean_string,
        default=True,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--town",
        type=str,
        default="current",
        help="Connection label only; connection.py keeps the currently loaded CARLA map.",
    )
    parser.add_argument(
        "--safe-spawns",
        type=parse_spawn_numbers,
        default=parse_spawn_numbers(DEFAULT_SAFE_SPAWNS),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(TOTAL_TIMESTEPS),
    )
    parser.add_argument(
        "--session-end-step",
        type=int,
        default=0,
        help=(
            "Stop THIS invocation at this absolute GLOBAL step while keeping "
            "--total-timesteps as the full training/curriculum horizon. "
            "0 means run until --total-timesteps."
        ),
    )
    parser.add_argument(
        "--test-timesteps",
        type=int,
        default=int(TEST_TIMESTEPS),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=int(PPO_ROLLOUT_STEPS),
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=float(PPO_MAX_EPISODE_SECONDS),
    )

    parser.add_argument(
        "--action-std-init",
        type=float,
        default=float(ACTION_STD_INIT),
    )
    parser.add_argument(
        "--action-std-min",
        type=float,
        default=float(PPO_ACTION_STD_MIN),
    )
    parser.add_argument(
        "--action-std-decay",
        type=float,
        default=float(PPO_ACTION_STD_DECAY),
    )
    parser.add_argument(
        "--action-std-decay-freq",
        type=int,
        default=int(PPO_ACTION_STD_DECAY_FREQ),
    )

    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=int(PPO_CHECKPOINT_EVERY_STEPS),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=boolean_string,
        default=bool(MODEL_LOAD),
    )

    parser.add_argument(
        "--curriculum",
        type=boolean_string,
        default=False,
    )
    parser.add_argument(
        "--desired-speed",
        type=float,
        default=0.35,
        help="Right-turn safe baseline target speed [m/s].",
    )

    parser.add_argument(
        "--ppo-device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--encoder-device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
    )

    parser.add_argument(
        "--tensorboard",
        type=boolean_string,
        default=True,
    )

    parser.add_argument(
        "--print-every-steps",
        type=int,
        default=50,
        help=(
            "Verbose terminal monitor interval in environment steps. "
            "Use 0 to disable."
        ),
    )

    parser.add_argument(
        "--policy-speed-cap",
        type=float,
        default=0.35,
        help="Hard execution cap for policy speed command [m/s].",
    )
    parser.add_argument(
        "--dynamics-dr",
        type=boolean_string,
        default=False,
        help="Enable V3 dynamics DR. Keep false for nominal baseline.",
    )
    parser.add_argument(
        "--camera-pose-dr",
        type=boolean_string,
        default=False,
    )
    parser.add_argument(
        "--weather-dr",
        type=boolean_string,
        default=False,
    )
    parser.add_argument(
        "--image-dr",
        type=boolean_string,
        default=False,
    )
    parser.add_argument(
        "--sensor-noise-dr",
        type=boolean_string,
        default=False,
    )

    return parser.parse_args()


def validate_args(args):
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be > 0.")

    if args.session_end_step < 0:
        raise ValueError("--session-end-step must be >= 0.")

    if (
        args.session_end_step > 0
        and args.session_end_step > args.total_timesteps
    ):
        raise ValueError(
            "--session-end-step must be <= --total-timesteps."
        )

    if args.test_timesteps <= 0:
        raise ValueError("--test-timesteps must be > 0.")

    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be > 0.")

    if args.max_episode_seconds <= 0.0:
        raise ValueError("--max-episode-seconds must be > 0.")

    if args.action_std_init <= 0.0:
        raise ValueError("--action-std-init must be > 0.")

    if args.action_std_min <= 0.0:
        raise ValueError("--action-std-min must be > 0.")

    if args.action_std_decay < 0.0:
        raise ValueError("--action-std-decay must be >= 0.")

    if args.action_std_decay_freq <= 0:
        raise ValueError("--action-std-decay-freq must be > 0.")

    if args.checkpoint_every_steps <= 0:
        raise ValueError("--checkpoint-every-steps must be > 0.")

    if args.print_every_steps < 0:
        raise ValueError("--print-every-steps must be >= 0.")

    if not (0.0 <= float(args.desired_speed) <= 1.0):
        raise ValueError("--desired-speed must be in [0,1] m/s.")

    if not (0.0 < float(args.policy_speed_cap) <= 1.0):
        raise ValueError("--policy-speed-cap must be in (0,1] m/s.")

    if float(args.desired_speed) > float(args.policy_speed_cap) + 1e-12:
        raise ValueError(
            "--desired-speed must be <= --policy-speed-cap."
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def curriculum_speed(global_step, total_timesteps):
    """
    PPO_CURRICULUM items are:
        (fraction_of_total_training, desired_speed_mps)
    """
    total_timesteps = max(int(total_timesteps), 1)
    progress = float(global_step) / float(total_timesteps)

    selected_speed = float(PPO_CURRICULUM[0][1])
    selected_stage = 0

    for stage_index, (fraction, speed_mps) in enumerate(PPO_CURRICULUM):
        if progress + 1e-12 >= float(fraction):
            selected_speed = float(speed_mps)
            selected_stage = int(stage_index)
        else:
            break

    return selected_stage, selected_speed


def current_task_speed(args, global_step):
    if args.curriculum:
        return curriculum_speed(
            global_step=global_step,
            total_timesteps=args.total_timesteps,
        )

    return -1, float(args.desired_speed)


def trainer_state_path(agent):
    return os.path.join(
        agent._checkpoint_dir(),
        "training_state_v3.json",
    )


def save_training_state(
    agent,
    global_step,
    episode,
    curriculum_stage,
    desired_speed_mps,
    checkpoint_path,
):
    path = trainer_state_path(agent)

    data = {
        "version": "TRAINER_V3_GD15",
        "global_step": int(global_step),
        "episode": int(episode),
        "curriculum_stage": int(curriculum_stage),
        "desired_speed_mps": float(desired_speed_mps),
        "action_std": float(agent.action_std),
        "checkpoint_path": str(checkpoint_path),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )

    return path


def load_training_state(agent):
    path = trainer_state_path(agent)

    if not os.path.isfile(path):
        return None

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if data.get("version") != "TRAINER_V3_GD15":
        raise RuntimeError(
            "Training state is not GĐ15 V3: {}".format(path)
        )

    return data


def save_checkpoint(
    agent,
    global_step,
    episode,
    curriculum_stage,
    desired_speed_mps,
):
    checkpoint_path = agent.save()

    state_path = save_training_state(
        agent=agent,
        global_step=global_step,
        episode=episode,
        curriculum_stage=curriculum_stage,
        desired_speed_mps=desired_speed_mps,
        checkpoint_path=checkpoint_path,
    )

    print(
        "CHECKPOINT | step={} | policy={} | state={}".format(
            global_step,
            checkpoint_path,
            state_path,
        )
    )

    return checkpoint_path


def make_writer(args):
    if not args.tensorboard:
        return None

    if SummaryWriter is None:
        print("WARNING: TensorBoard unavailable; continuing without it.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "train" if args.train else "eval"

    run_dir = os.path.join(
        "runs",
        "{}_{}_{}".format(
            args.model_name,
            mode,
            timestamp,
        ),
    )

    writer = SummaryWriter(run_dir)

    writer.add_text(
        "config",
        "\n".join(
            "{} = {}".format(key, value)
            for key, value in sorted(vars(args).items())
        ),
    )

    print("TensorBoard:", run_dir)

    return writer


# VERBOSE_TRAIN_MONITOR_V1
def _safe_float(value, default=float("nan")):
    try:
        return float(value)
    except Exception:
        return float(default)


def _fmt(value, pattern):
    value = _safe_float(value)
    try:
        return pattern.format(value)
    except Exception:
        return str(value)


def print_verbose_step_v1(
    global_step,
    total_timesteps,
    episode,
    action,
    reward,
    info,
    agent,
    wall_start_time,
):
    imu = info.get("imu", {}) or {}
    control = info.get("control", {}) or {}
    domain = info.get("domain_randomization", {}) or {}

    elapsed_s = max(
        1e-9,
        time.time() - float(wall_start_time),
    )
    sps = float(global_step) / elapsed_s

    remaining = max(
        0,
        int(total_timesteps) - int(global_step),
    )
    eta_s = (
        float(remaining) / sps
        if sps > 1e-9
        else float("nan")
    )

    progress_pct = (
        100.0
        * float(global_step)
        / max(1, int(total_timesteps))
    )

    logical_steer = control.get(
        "actual_steer_cmd",
        action[0],
    )
    logical_speed = control.get(
        "actual_speed_cmd_mps",
        action[1],
    )

    physical_steer = control.get(
        "gd6_physical_steer_cmd",
        control.get(
            "actual_steer_cmd",
            action[0],
        ),
    )

    throttle = control.get(
        "throttle",
        float("nan"),
    )
    brake = control.get(
        "brake",
        float("nan"),
    )

    memory_size = len(
        getattr(
            getattr(agent, "memory", None),
            "rewards",
            [],
        )
    )

    print(
        "\n"
        "[LIVE] step={}/{} ({:.2f}%) | ep={} | rollout={} | "
        "SPS={:.2f} | ETA={:.1f} min\n"
        "  PPO action      : steer={:+.4f} | speed_cmd={:.4f} m/s\n"
        "  Applied logical : steer={:+.4f} | speed_cmd={:.4f} m/s\n"
        "  CARLA physical  : steer={:+.4f} | throttle={} | brake={}\n"
        "  IMU             : speed={} m/s | yaw={} rad/s | ax={} m/s^2\n"
        "  Tracking        : lat={} m | heading={} rad | progress={} m\n"
        "  Reward          : {:+.6f} | done={} | term={} | trunc={} | reason={}\n"
        "  DR              : torque={} | brake={} | gain+={} | gain-={} | delay={} tick\n"
        "  PPO             : action_std={:.4f} | sim_frame={}".format(
            int(global_step),
            int(total_timesteps),
            progress_pct,
            int(episode),
            int(memory_size),
            sps,
            eta_s / 60.0,
            _safe_float(action[0]),
            _safe_float(action[1]),
            _safe_float(logical_steer),
            _safe_float(logical_speed),
            _safe_float(physical_steer),
            _fmt(throttle, "{:.4f}"),
            _fmt(brake, "{:.4f}"),
            _fmt(
                imu.get("speed_mps"),
                "{:.4f}",
            ),
            _fmt(
                imu.get("yaw_rate_rad_s"),
                "{:+.5f}",
            ),
            _fmt(
                imu.get(
                    "longitudinal_accel_mps2"
                ),
                "{:+.5f}",
            ),
            _fmt(
                info.get("lateral_error_m"),
                "{:+.5f}",
            ),
            _fmt(
                info.get("heading_error_rad"),
                "{:+.5f}",
            ),
            _fmt(
                info.get("forward_progress_m"),
                "{:+.5f}",
            ),
            _safe_float(reward),
            bool(info.get("done", False)),
            bool(info.get("terminated", False)),
            bool(info.get("truncated", False)),
            info.get("termination_reason"),
            _fmt(
                domain.get("torque_scale"),
                "{:.4f}",
            ),
            _fmt(
                domain.get("max_brake_torque"),
                "{:.1f}",
            ),
            _fmt(
                domain.get(
                    "steer_gain_positive"
                ),
                "{:.4f}",
            ),
            _fmt(
                domain.get(
                    "steer_gain_negative"
                ),
                "{:.4f}",
            ),
            domain.get(
                "extra_command_delay_ticks",
                "?",
            ),
            _safe_float(agent.action_std),
            info.get("sim_frame"),
        )
    )


class EpisodeStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.reward = 0.0
        self.steps = 0

        self.speeds = []
        self.steers = []
        self.speed_cmds = []
        self.lateral_errors = []
        self.heading_errors = []
        self.progress = []

        self.reason = None
        self.terminated = False
        self.truncated = False

    def add(self, action, reward, info):
        self.reward += float(reward)
        self.steps += 1

        self.speeds.append(
            float(info["imu"]["speed_mps"])
        )
        self.steers.append(
            float(action[0])
        )
        self.speed_cmds.append(
            float(action[1])
        )
        self.lateral_errors.append(
            float(info["lateral_error_m"])
        )
        self.heading_errors.append(
            float(info["heading_error_rad"])
        )
        self.progress.append(
            float(info["forward_progress_m"])
        )

        if info["done"]:
            self.reason = info["termination_reason"]
            self.terminated = bool(info["terminated"])
            self.truncated = bool(info["truncated"])

    @staticmethod
    def _mean(values):
        array = np.asarray(values)

        if array.size == 0:
            return 0.0

        return float(np.mean(array))

    @staticmethod
    def _max_abs(values):
        array = np.asarray(values)

        if array.size == 0:
            return 0.0

        return float(np.max(np.abs(array)))

    def summary(self):
        return {
            "reward": float(self.reward),
            "steps": int(self.steps),
            "avg_speed_mps": self._mean(self.speeds),
            "max_speed_mps": float(max(self.speeds)) if self.speeds else 0.0,
            "avg_abs_steer": self._mean(np.abs(self.steers)),
            "avg_speed_cmd_mps": self._mean(self.speed_cmds),
            "avg_abs_lateral_m": self._mean(np.abs(self.lateral_errors)),
            "max_abs_lateral_m": self._max_abs(self.lateral_errors),
            "avg_abs_heading_rad": self._mean(np.abs(self.heading_errors)),
            "progress_m": float(np.sum(self.progress)) if self.progress else 0.0,
            "reason": self.reason,
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
        }


def print_episode(
    episode,
    global_step,
    total_timesteps,
    desired_speed_mps,
    action_std,
    summary,
):
    print(
        "\n"
        "EPISODE {:05d} | global_step={}/{}\n"
        "  reason={} | terminated={} | truncated={}\n"
        "  steps={} | reward={:+.3f} | progress={:+.3f} m\n"
        "  desired_speed={:.3f} | avg_speed={:.3f} | max_speed={:.3f} m/s\n"
        "  avg_speed_cmd={:.3f} | avg|steer|={:.3f}\n"
        "  avg|lat|={:.4f} m | max|lat|={:.4f} m | avg|heading|={:.4f} rad\n"
        "  action_std={:.4f}".format(
            episode,
            global_step,
            total_timesteps,
            summary["reason"],
            summary["terminated"],
            summary["truncated"],
            summary["steps"],
            summary["reward"],
            summary["progress_m"],
            desired_speed_mps,
            summary["avg_speed_mps"],
            summary["max_speed_mps"],
            summary["avg_speed_cmd_mps"],
            summary["avg_abs_steer"],
            summary["avg_abs_lateral_m"],
            summary["max_abs_lateral_m"],
            summary["avg_abs_heading_rad"],
            action_std,
        )
    )


def write_episode(writer, episode, global_step, summary, desired_speed_mps):
    if writer is None:
        return

    writer.add_scalar(
        "episode/reward",
        summary["reward"],
        global_step,
    )
    writer.add_scalar(
        "episode/steps",
        summary["steps"],
        global_step,
    )
    writer.add_scalar(
        "episode/progress_m",
        summary["progress_m"],
        global_step,
    )
    writer.add_scalar(
        "episode/avg_speed_mps",
        summary["avg_speed_mps"],
        global_step,
    )
    writer.add_scalar(
        "episode/max_speed_mps",
        summary["max_speed_mps"],
        global_step,
    )
    writer.add_scalar(
        "episode/avg_abs_lateral_m",
        summary["avg_abs_lateral_m"],
        global_step,
    )
    writer.add_scalar(
        "episode/max_abs_lateral_m",
        summary["max_abs_lateral_m"],
        global_step,
    )
    writer.add_scalar(
        "episode/avg_abs_heading_rad",
        summary["avg_abs_heading_rad"],
        global_step,
    )
    writer.add_scalar(
        "episode/avg_abs_steer",
        summary["avg_abs_steer"],
        global_step,
    )
    writer.add_scalar(
        "task/desired_speed_mps",
        desired_speed_mps,
        global_step,
    )
    writer.add_scalar(
        "episode/index",
        episode,
        global_step,
    )


def write_update(writer, global_step, agent, metrics):
    if writer is None or metrics is None:
        return

    for key, value in metrics.items():
        writer.add_scalar(
            "ppo/{}".format(key),
            float(value),
            global_step,
        )

    writer.add_scalar(
        "ppo/action_std",
        float(agent.action_std),
        global_step,
    )


def maybe_decay_action_std(
    agent,
    global_step,
    previous_bucket,
    args,
):
    bucket = int(
        global_step // args.action_std_decay_freq
    )

    if bucket <= previous_bucket:
        return previous_bucket

    for _ in range(
        bucket - previous_bucket
    ):
        if agent.action_std <= args.action_std_min + 1e-12:
            break

        agent.decay_action_std(
            action_std_decay_rate=args.action_std_decay,
            min_action_std=args.action_std_min,
        )

    print(
        "ACTION STD | step={} | std={:.4f}".format(
            global_step,
            agent.action_std,
        )
    )

    return bucket


def train(args, env, agent, writer):
    global_step = 0
    episode = 0

    if args.load_checkpoint:
        checkpoint = agent.load()
        state = load_training_state(agent)

        print("Loaded PPO:", checkpoint)

        if state is not None:
            global_step = int(
                state.get("global_step", 0)
            )
            episode = int(
                state.get("episode", 0)
            )

            print(
                "Loaded trainer state | step={} | episode={}".format(
                    global_step,
                    episode,
                )
            )
        else:
            print(
                "WARNING: no training_state_v3.json; "
                "resume step starts at 0."
            )

    session_end_step = (
        int(args.session_end_step)
        if int(args.session_end_step) > 0
        else int(args.total_timesteps)
    )

    if session_end_step <= global_step:
        raise RuntimeError(
            "session_end_step={} must be > loaded global_step={}.".format(
                session_end_step,
                global_step,
            )
        )

    stage, desired_speed = current_task_speed(
        args,
        global_step,
    )
    env.set_desired_speed_mps(
        desired_speed
    )

    decay_bucket = int(
        global_step // args.action_std_decay_freq
    )

    next_checkpoint_step = (
        (
            global_step
            // args.checkpoint_every_steps
        )
        + 1
    ) * args.checkpoint_every_steps

    observation = env.reset()
    stats = EpisodeStats()

    print(
        "\nTRAIN START | model={} | step={}/{} | session_end={} | rollout={} | "
        "desired_speed={:.3f} | curriculum={}".format(
            args.model_name,
            global_step,
            args.total_timesteps,
            session_end_step,
            args.rollout_steps,
            desired_speed,
            args.curriculum,
        )
    )

    last_info = None
    verbose_wall_start = time.time()

    while global_step < session_end_step:
        new_stage, new_speed = current_task_speed(
            args,
            global_step,
        )

        if (
            new_stage != stage
            or abs(new_speed - desired_speed) > 1e-12
        ):
            stage = new_stage
            desired_speed = new_speed
            env.set_desired_speed_mps(
                desired_speed
            )

            print(
                "\nCURRICULUM | stage={} | step={} | desired_speed={:.3f} m/s".format(
                    stage,
                    global_step,
                    desired_speed,
                )
            )

        action = agent.get_action(
            observation,
            train=True,
        )

        try:
            (
                next_observation,
                reward,
                done,
                info,
            ) = env.step(action)

        except Exception as env_error:
            # get_action(train=True) has already appended one pending
            # transition to the PPO buffer. Because env.step() failed,
            # that transition has no reward/outcome and must never be
            # used by PPO. Drop the current partial rollout before
            # writing an emergency checkpoint.
            pending_observations = len(agent.memory.observation)
            pending_rewards = len(agent.memory.rewards)
            agent.memory.clear()

            print(
                "\nENV FAILURE | step={} | episode={} | "
                "discarded_rollout_obs={} | discarded_rewards={} | "
                "error={}".format(
                    global_step,
                    episode,
                    pending_observations,
                    pending_rewards,
                    repr(env_error),
                )
            )

            try:
                emergency_path = save_checkpoint(
                    agent=agent,
                    global_step=global_step,
                    episode=episode,
                    curriculum_stage=stage,
                    desired_speed_mps=desired_speed,
                )
                print(
                    "EMERGENCY CHECKPOINT OK | {}".format(
                        emergency_path
                    )
                )
            except Exception:
                logging.exception(
                    "Emergency checkpoint failed after env error."
                )

            if writer is not None:
                try:
                    writer.flush()
                except Exception:
                    pass

            raise

        agent.record_outcome(
            reward=reward,
            terminated=info["terminated"],
            truncated=info["truncated"],
            next_obs=(
                next_observation
                if info["truncated"]
                else None
            ),
        )

        stats.add(
            action=action,
            reward=reward,
            info=info,
        )

        global_step += 1
        last_info = info

        if (
            args.print_every_steps > 0
            and (
                global_step % args.print_every_steps == 0
                or done
            )
        ):
            print_verbose_step_v1(
                global_step=global_step,
                total_timesteps=args.total_timesteps,
                episode=episode + (1 if done else 0),
                action=action,
                reward=reward,
                info=info,
                agent=agent,
                wall_start_time=verbose_wall_start,
            )

        decay_bucket = maybe_decay_action_std(
            agent=agent,
            global_step=global_step,
            previous_bucket=decay_bucket,
            args=args,
        )

        update_due = (
            len(agent.memory.rewards)
            >= args.rollout_steps
            or global_step
            >= session_end_step
        )

        if update_due:
            boundary = bool(
                info["terminated"]
                or info["truncated"]
            )

            metrics = agent.learn(
                last_obs=(
                    None
                    if boundary
                    else next_observation
                )
            )

            print(
                "PPO UPDATE | step={} | loss={:+.6f} | "
                "policy={:+.6f} | value={:+.6f} | "
                "entropy={:+.6f}".format(
                    global_step,
                    metrics["loss"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy"],
                )
            )

            print(
                "PPO DETAIL | step={} | mean_return={:+.6f} | "
                "mean_advantage={:+.6f} | action_std={:.4f}".format(
                    global_step,
                    float(
                        metrics.get(
                            "mean_return",
                            float("nan"),
                        )
                    ),
                    float(
                        metrics.get(
                            "mean_advantage",
                            float("nan"),
                        )
                    ),
                    float(agent.action_std),
                )
            )

            write_update(
                writer=writer,
                global_step=global_step,
                agent=agent,
                metrics=metrics,
            )

        if global_step >= next_checkpoint_step:
            save_checkpoint(
                agent=agent,
                global_step=global_step,
                episode=episode,
                curriculum_stage=stage,
                desired_speed_mps=desired_speed,
            )

            while global_step >= next_checkpoint_step:
                next_checkpoint_step += (
                    args.checkpoint_every_steps
                )

        if done:
            episode += 1
            summary = stats.summary()

            print_episode(
                episode=episode,
                global_step=global_step,
                total_timesteps=args.total_timesteps,
                desired_speed_mps=desired_speed,
                action_std=agent.action_std,
                summary=summary,
            )

            write_episode(
                writer=writer,
                episode=episode,
                global_step=global_step,
                summary=summary,
                desired_speed_mps=desired_speed,
            )

            stats.reset()

            if global_step < session_end_step:
                observation = env.reset()
            else:
                observation = next_observation

        else:
            observation = next_observation

    # Always keep a final resumable policy.
    save_checkpoint(
        agent=agent,
        global_step=global_step,
        episode=episode,
        curriculum_stage=stage,
        desired_speed_mps=desired_speed,
    )

    if stats.steps > 0:
        partial = stats.summary()
        partial["reason"] = (
            "training_limit"
            if global_step >= int(args.total_timesteps)
            else "session_limit"
        )

        print_episode(
            episode=episode + 1,
            global_step=global_step,
            total_timesteps=args.total_timesteps,
            desired_speed_mps=desired_speed,
            action_std=agent.action_std,
            summary=partial,
        )

    if global_step >= int(args.total_timesteps):
        print("\nTRAIN COMPLETE | steps={}".format(global_step))
    else:
        print(
            "\nSESSION COMPLETE | steps={} | full_target={}".format(
                global_step,
                args.total_timesteps,
            )
        )


def evaluate(args, env, agent, writer):
    if not args.load_checkpoint:
        raise RuntimeError(
            "Evaluation requires --load-checkpoint true."
        )

    checkpoint = agent.load()
    print("Loaded PPO:", checkpoint)

    desired_speed = float(
        args.desired_speed
    )
    env.set_desired_speed_mps(
        desired_speed
    )

    timestep = 0
    episode = 0

    observation = env.reset()
    stats = EpisodeStats()

    print(
        "\nEVAL START | model={} | timesteps={} | desired_speed={:.3f}".format(
            args.model_name,
            args.test_timesteps,
            desired_speed,
        )
    )

    while timestep < args.test_timesteps:
        action = agent.get_action(
            observation,
            train=False,
        )

        (
            next_observation,
            reward,
            done,
            info,
        ) = env.step(action)

        stats.add(
            action=action,
            reward=reward,
            info=info,
        )

        timestep += 1

        if done:
            episode += 1
            summary = stats.summary()

            print_episode(
                episode=episode,
                global_step=timestep,
                total_timesteps=args.test_timesteps,
                desired_speed_mps=desired_speed,
                action_std=agent.action_std,
                summary=summary,
            )

            write_episode(
                writer=writer,
                episode=episode,
                global_step=timestep,
                summary=summary,
                desired_speed_mps=desired_speed,
            )

            stats.reset()

            if timestep < args.test_timesteps:
                observation = env.reset()
        else:
            observation = next_observation

    print(
        "\nEVAL COMPLETE | episodes={} | steps={}".format(
            episode,
            timestep,
        )
    )


def runner():
    args = parse_args()
    validate_args(args)

    seed_everything(
        args.seed
    )

    writer = make_writer(
        args
    )

    env = None

    try:
        client, world = (
            ClientConnection(
                args.town
            ).setup()
        )

        if client is None or world is None:
            raise RuntimeError(
                "CARLA connection failed."
            )

        env = CarlaEnvironmentRGB(
            client=client,
            world=world,
            town=args.town,
            safe_spawn_numbers=args.safe_spawns,
            desired_speed_mps=float(
                args.desired_speed
            ),
            max_episode_seconds=float(
                args.max_episode_seconds
            ),
            encoder_device=args.encoder_device,
            # SIM-TO-REAL SAFE ABLATION SWITCHES
            domain_randomization_enabled=bool(
                args.train and args.dynamics_dr
            ),
            domain_randomization_seed=int(args.seed),
            policy_speed_cap_mps=float(args.policy_speed_cap),
            camera_pose_dr_enabled=bool(
                args.train and args.camera_pose_dr
            ),
            weather_dr_enabled=bool(
                args.train and args.weather_dr
            ),
            image_dr_enabled=bool(
                args.train and args.image_dr
            ),
            sensor_noise_dr_enabled=bool(
                args.train and args.sensor_noise_dr
            ),
        )

        agent = PPOAgent(
            town=args.model_name,
            action_std_init=float(
                args.action_std_init
            ),
            device=args.ppo_device,
        )

        if args.train:
            train(
                args=args,
                env=env,
                agent=agent,
                writer=writer,
            )
        else:
            evaluate(
                args=args,
                env=env,
                agent=agent,
                writer=writer,
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception:
        logging.exception(
            "GĐ15 trainer failed."
        )
        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        if writer is not None:
            writer.flush()
            writer.close()

        print("Đã cleanup.")


if __name__ == "__main__":
    runner()
