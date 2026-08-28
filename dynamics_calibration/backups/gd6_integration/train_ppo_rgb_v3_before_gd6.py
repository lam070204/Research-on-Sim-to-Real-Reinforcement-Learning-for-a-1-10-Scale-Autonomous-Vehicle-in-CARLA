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
from simulation.carla_environment_rgb_v3 import CarlaEnvironmentRGB

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


DEFAULT_MODEL_NAME = "automav3_rgb_v3"
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
        default=True,
    )
    parser.add_argument(
        "--desired-speed",
        type=float,
        default=0.60,
        help="Used when --curriculum false and for evaluation.",
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

    return parser.parse_args()


def validate_args(args):
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be > 0.")

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

    if not (0.0 <= float(args.desired_speed) <= 1.0):
        raise ValueError("--desired-speed must be in [0,1] m/s.")


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
        "\nTRAIN START | model={} | step={}/{} | rollout={} | "
        "desired_speed={:.3f} | curriculum={}".format(
            args.model_name,
            global_step,
            args.total_timesteps,
            args.rollout_steps,
            desired_speed,
            args.curriculum,
        )
    )

    last_info = None

    while global_step < args.total_timesteps:
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
            >= args.total_timesteps
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

            if global_step < args.total_timesteps:
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
        partial["reason"] = "training_limit"

        print_episode(
            episode=episode + 1,
            global_step=global_step,
            total_timesteps=args.total_timesteps,
            desired_speed_mps=desired_speed,
            action_std=agent.action_std,
            summary=partial,
        )

    print("\nTRAIN COMPLETE | steps={}".format(global_step))


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
