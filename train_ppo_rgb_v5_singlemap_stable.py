# -*- coding: utf-8 -*-
"""
PPO V5 SINGLE-MAP STABLE035
===========================

One shared PPO policy, two CARLA workers, SAME map.

Design:
- map: /Game/mapvuong/mapvuong
- obs100 unchanged: latent95 + speed + yaw + ax + prevSteer + prevSpeed
- action2 unchanged: steer + normalized speed
- hard speed max defaults to 0.35 m/s
- speed reward follows original repo ratios: 15/25, 22/25, 25/25
- recovery injection OFF
- spawn perturbation OFF
- command stabilization: deadband + steer/speed slew before Dynamics DR
- controlled DR schedule; at most one DR family per randomized episode
- failure-location clustering is diagnostic only; never auto-oversampled
- checkpoint every configurable number of transitions
- correct PPO terminal/truncation handling inherited from PPOAgent V5

Python 3.7+.
"""

from __future__ import print_function

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import re
import time
import traceback
from datetime import datetime

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from networks.on_policy.ppo.ppo_agent_v5 import PPOAgent
from parameters_v5 import (
    ACTION_STD_INIT,
    PPO_ACTION_STD_DECAY,
    PPO_ACTION_STD_DECAY_FREQ,
    PPO_ACTION_STD_MIN,
    PPO_CHECKPOINT_DIR,
    PPO_CHECKPOINT_EVERY_STEPS,
    PPO_MAX_EPISODE_SECONDS,
    PPO_ROLLOUT_STEPS,
    SEED,
    TOTAL_TIMESTEPS,
)
from simulation.carla_connection_v5 import carla
from simulation.carla_environment_rgb_v5_singlemap_stable import (
    CarlaEnvironmentRGBV5SingleMapStable as CarlaEnvironmentRGBV5,
    configure_recovery_scenarios_v5,
)


MODEL_NAME = "automav5_rgb_mapvuong_stable035_v14_lookahead"
STATE_NAME = "training_state_mapvuong_stable035.json"
TRAINER_VERSION = "TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_7_ORACLE_TEACHER"
COMPATIBLE_STATE_VERSIONS = {
    "TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_4_CORNER_LOOKAHEAD",
    "TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_4_1_PROBEFIX",
    "TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_4_2_PROBECOMPAT",
    "TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_5_LOWNOISE_CONTINUE",
    "TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_5_1_STARTUPFIX",
    "TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_7_ORACLE_TEACHER",
}

EXPECTED_SPEED_PROFILE = {
    "min_ratio": 15.0 / 25.0,
    "target_ratio": 22.0 / 25.0,
}

STATE_NAMES = (
    "stable",
    "mild",
    "moderate",
    "strong",
)

REWARD_TERM_NAMES = (
    "step",
    "progress",
    "lane",
    "heading",
    "speed_score",
    "speed_cmd_target",
    "measured_speed_target",
    "recovery_lateral_progress",
    "recovery_heading_progress",
    "recovery_success_bonus",
    "steer_smooth",
    "speed_cmd_smooth",
    "collision",
    "offroad",
    "stuck",
)


def boolean_string(value):
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in ("true", "1", "yes", "y", "on"):
        return True

    if value in ("false", "0", "no", "n", "off"):
        return False

    raise argparse.ArgumentTypeError(
        "Use true/false."
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
            "At least one spawn is required."
        )

    return list(dict.fromkeys(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO V5 STABLE035 from step 0 "
            "with one or two independent CARLA workers."
        )
    )

    # --------------------------------------------------------------
    # CARLA workers
    # --------------------------------------------------------------
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2),
        default=2,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port0",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--port1",
        type=int,
        default=2010,
    )
    parser.add_argument(
        "--expected-map",
        type=str,
        default="mapvuong",
        help=(
            "Substring that must occur in world.get_map().name."
        ),
    )
    parser.add_argument(
        "--carla-timeout",
        type=float,
        default=120.0,
    )

    # --------------------------------------------------------------
    # Fresh training / resume
    # --------------------------------------------------------------
    parser.add_argument(
        "--resume",
        type=boolean_string,
        default=False,
        help=(
            "False = strict step-0 fresh start. "
            "True = resume only this STABLE035 namespace."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help=(
            "Optional PPO_V5_CLEAN checkpoint used only to initialize a NEW "
            "one-spawn run at global_step=0. Cannot be combined with --resume true."
        ),
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(TOTAL_TIMESTEPS),
    )
    parser.add_argument(
        "--rollout-total",
        type=int,
        default=1024,
        help=(
            "Total transitions per PPO update across all workers. "
            "ONE-SPAWN default is 1024 for faster feedback."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=int(PPO_CHECKPOINT_EVERY_STEPS),
    )

    # --------------------------------------------------------------
    # Task
    # --------------------------------------------------------------
    parser.add_argument(
        "--max-speed",
        dest="desired_speed",
        type=float,
        default=0.35,
        help=(
            "Hard speed ceiling. Original-style reward derives "
            "MIN=15/25*MAX and TARGET=22/25*MAX."
        ),
    )
    parser.add_argument(
        "--safe-spawns",
        type=parse_spawn_numbers,
        default=parse_spawn_numbers("1,2,3,4"),
        help="Balanced spawn numbers on this ONE map, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=float(
            PPO_MAX_EPISODE_SECONDS
        ),
    )

    # --------------------------------------------------------------
    # ONE-SPAWN delayed recovery curriculum
    # --------------------------------------------------------------
    parser.add_argument(
        "--disturbance-prob",
        type=float,
        default=0.0,
        help="Recovery injection is OFF in STABLE035 baseline.",
    )
    parser.add_argument(
        "--disturbance-duration-min-s",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--disturbance-duration-max-s",
        type=float,
        default=0.16,
    )
    parser.add_argument(
        "--disturbance-steer-min",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--disturbance-steer-max",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--recovery-min-episode-step",
        type=int,
        default=700,
        help="No injected recovery before this episode tick (700 ticks = 14 s at 50 Hz).",
    )
    parser.add_argument(
        "--recovery-stable-ticks",
        type=int,
        default=100,
        help="Require this many consecutive stable ticks before injection.",
    )
    parser.add_argument(
        "--recovery-stable-lateral-m",
        type=float,
        default=0.040,
    )
    parser.add_argument(
        "--recovery-stable-heading-deg",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--recovery-stable-speed-mps",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--recovery-cooldown-ticks",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--recovery-max-events",
        type=int,
        default=1,
        help="Start with at most one injected recovery event per episode.",
    )

    # --------------------------------------------------------------
    # Sim-to-real DR switches
    # --------------------------------------------------------------
    parser.add_argument(
        "--dynamics-dr",
        type=boolean_string,
        default=True,
    )
    parser.add_argument(
        "--vision-dr",
        type=boolean_string,
        default=True,
    )
    parser.add_argument(
        "--sensor-dr",
        type=boolean_string,
        default=True,
    )
    parser.add_argument(
        "--recovery-scenarios",
        type=boolean_string,
        default=False,
    )

    # --------------------------------------------------------------
    # Stability controls / controlled DR schedule
    # --------------------------------------------------------------
    parser.add_argument("--steer-rate-limit", type=float, default=3.0,
                        help="Max steering command slew per second.")
    parser.add_argument("--steer-deadband", type=float, default=0.015,
                        help="Neutral deadband to suppress small servo chatter.")
    parser.add_argument("--speed-rate-limit", type=float, default=0.50,
                        help="Target-speed slew limit in m/s^2.")
    parser.add_argument("--dr-family-budget", type=int, default=1,
                        help="At most this many DR families per randomized episode.")

    parser.add_argument("--phase-clean-until", type=int, default=100000)
    parser.add_argument("--phase-vision-until", type=int, default=300000)
    parser.add_argument("--phase-dynamics-until", type=int, default=600000)
    parser.add_argument("--phase-controlled-until", type=int, default=900000)
    parser.add_argument("--dr-prob-vision", type=float, default=0.25)
    parser.add_argument("--dr-prob-dynamics", type=float, default=0.35)
    parser.add_argument("--dr-prob-controlled", type=float, default=0.45)
    parser.add_argument("--dr-prob-polish", type=float, default=0.20)

    # --------------------------------------------------------------
    # Seeds / devices
    # --------------------------------------------------------------
    parser.add_argument(
        "--seed0",
        type=int,
        default=505,
    )
    parser.add_argument(
        "--seed1",
        type=int,
        default=1505,
    )

    parser.add_argument(
        "--learner-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--worker-ppo-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--encoder-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--worker-torch-threads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--learner-torch-threads",
        type=int,
        default=2,
    )

    # --------------------------------------------------------------
    # Exploration schedule
    # --------------------------------------------------------------
    parser.add_argument(
        "--action-std-init",
        type=float,
        default=float(
            ACTION_STD_INIT
        ),
    )
    parser.add_argument(
        "--action-std-min",
        type=float,
        default=float(
            PPO_ACTION_STD_MIN
        ),
    )
    parser.add_argument(
        "--action-std-decay",
        type=float,
        default=float(
            PPO_ACTION_STD_DECAY
        ),
    )
    parser.add_argument(
        "--action-std-decay-freq",
        type=int,
        default=int(
            PPO_ACTION_STD_DECAY_FREQ
        ),
    )

    # --------------------------------------------------------------
    # Logging / smoke
    # --------------------------------------------------------------
    parser.add_argument(
        "--tensorboard",
        type=boolean_string,
        default=True,
    )
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--smoke-test",
        type=boolean_string,
        default=False,
        help=(
            "Collect a short rollout and exit. "
            "No PPO update, checkpoint, or state mutation."
        ),
    )
    parser.add_argument(
        "--smoke-steps-per-worker",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--probe-action-std",
        type=float,
        default=-1.0,
        help=(
            "Smoke/probe only. If > 0, override the loaded action std "
            "for rollout collection without mutating training state. "
            "Use a tiny value such as 0.0001 to approximate deterministic "
            "actor-mean evaluation."
        ),
    )
    parser.add_argument(
        "--resume-action-std",
        type=float,
        default=-1.0,
        help=(
            "Resume-training only. If > 0, force PPO action std to this "
            "value after loading the checkpoint. Does not apply to smoke."
        ),
    )

    return parser.parse_args()


def validate_args(args):
    if int(args.workers) not in (1, 2):
        raise ValueError(
            "--workers must be 1 or 2."
        )

    if (
        int(args.workers) == 2
        and int(args.port0) == int(args.port1)
    ):
        raise ValueError(
            "port0 and port1 must differ for X2."
        )

    if bool(args.resume) and str(args.init_checkpoint).strip():
        raise ValueError(
            "Use either --resume true OR --init-checkpoint, not both."
        )

    if int(args.total_timesteps) <= 0:
        raise ValueError(
            "--total-timesteps must be > 0."
        )

    if int(args.rollout_total) < int(args.workers):
        raise ValueError(
            "--rollout-total must be >= worker count."
        )

    if int(args.checkpoint_every_steps) <= 0:
        raise ValueError(
            "--checkpoint-every-steps must be > 0."
        )

    if not (0.05 <= float(args.desired_speed) <= 0.35):
        raise ValueError(
            "--max-speed must be in [0.05, 0.35] m/s for STABLE035."
        )

    if int(args.dr_family_budget) < 1:
        raise ValueError("--dr-family-budget must be >= 1.")

    phase_points = [
        int(args.phase_clean_until),
        int(args.phase_vision_until),
        int(args.phase_dynamics_until),
        int(args.phase_controlled_until),
    ]
    if phase_points != sorted(phase_points):
        raise ValueError("DR phase boundaries must be nondecreasing.")

    for name in (
        "dr_prob_vision",
        "dr_prob_dynamics",
        "dr_prob_controlled",
        "dr_prob_polish",
    ):
        value = float(getattr(args, name))
        if not (0.0 <= value <= 1.0):
            raise ValueError("--{} must be in [0,1]".format(name.replace("_", "-")))

    if float(args.max_episode_seconds) <= 0.0:
        raise ValueError(
            "--max-episode-seconds must be > 0."
        )

    if (
        int(args.worker_torch_threads) <= 0
        or int(args.learner_torch_threads) <= 0
    ):
        raise ValueError(
            "Torch thread counts must be > 0."
        )

    if int(args.smoke_steps_per_worker) <= 0:
        raise ValueError(
            "--smoke-steps-per-worker must be > 0."
        )

    if (
        float(args.probe_action_std) == 0.0
        or float(args.probe_action_std) < -1.0
    ):
        raise ValueError(
            "--probe-action-std must be -1 (use loaded std) or > 0."
        )

    if (
        float(args.resume_action_std) == 0.0
        or float(args.resume_action_std) < -1.0
    ):
        raise ValueError(
            "--resume-action-std must be -1 (keep checkpoint std) or > 0."
        )

    if not (0.0 <= float(args.disturbance_prob) <= 1.0):
        raise ValueError(
            "--disturbance-prob must be in [0,1]."
        )

    if not (
        0.0 < float(args.disturbance_duration_min_s)
        <= float(args.disturbance_duration_max_s)
    ):
        raise ValueError(
            "Invalid disturbance duration range."
        )

    if not (
        0.0 <= float(args.disturbance_steer_min)
        <= float(args.disturbance_steer_max)
        <= 1.0
    ):
        raise ValueError(
            "Invalid disturbance steer range."
        )

    if int(args.recovery_min_episode_step) < 0:
        raise ValueError(
            "--recovery-min-episode-step must be >= 0."
        )

    if int(args.recovery_stable_ticks) <= 0:
        raise ValueError(
            "--recovery-stable-ticks must be > 0."
        )

    if float(args.recovery_stable_lateral_m) <= 0.0:
        raise ValueError(
            "--recovery-stable-lateral-m must be > 0."
        )

    if float(args.recovery_stable_heading_deg) <= 0.0:
        raise ValueError(
            "--recovery-stable-heading-deg must be > 0."
        )

    if float(args.recovery_stable_speed_mps) < 0.0:
        raise ValueError(
            "--recovery-stable-speed-mps must be >= 0."
        )

    if int(args.recovery_cooldown_ticks) < 0:
        raise ValueError(
            "--recovery-cooldown-ticks must be >= 0."
        )

    if int(args.recovery_max_events) <= 0:
        raise ValueError(
            "--recovery-max-events must be > 0."
        )

    if float(args.action_std_init) <= 0.0:
        raise ValueError(
            "--action-std-init must be > 0."
        )

    if float(args.action_std_min) <= 0.0:
        raise ValueError(
            "--action-std-min must be > 0."
        )

    if float(args.action_std_decay) < 0.0:
        raise ValueError(
            "--action-std-decay must be >= 0."
        )

    if int(args.action_std_decay_freq) <= 0:
        raise ValueError(
            "--action-std-decay-freq must be > 0."
        )


def seed_everything(seed):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_dir():
    directory = os.path.join(
        PPO_CHECKPOINT_DIR,
        MODEL_NAME,
    )

    os.makedirs(
        directory,
        exist_ok=True,
    )

    return directory


def state_path():
    return os.path.join(
        checkpoint_dir(),
        STATE_NAME,
    )


def policy_files():
    pattern = re.compile(
        r"^ppo_policy_(\d+)_\.pth$"
    )

    items = []

    for name in os.listdir(
        checkpoint_dir()
    ):
        match = pattern.match(name)

        if match is None:
            continue

        items.append(
            (
                int(
                    match.group(1)
                ),
                os.path.join(
                    checkpoint_dir(),
                    name,
                ),
            )
        )

    items.sort(
        key=lambda item: item[0]
    )

    return items


def failure_cluster_path():
    return os.path.join(
        checkpoint_dir(),
        "failure_clusters_stable035.json",
    )


class FailureClusterTracker(object):
    """
    Diagnostic only. It NEVER changes sampling probability or reward.

    This is deliberate: repeated failures are evidence to inspect, not a reason
    to let one bad location dominate the entire policy update.
    """
    def __init__(self, grid_size_carla_m=1.5):
        self.grid_size = float(grid_size_carla_m)
        self.cells = {}

    def _key(self, event):
        gx = int(round(float(event.get("world_x", 0.0)) / self.grid_size))
        gy = int(round(float(event.get("world_y", 0.0)) / self.grid_size))
        return "{}|{}".format(gx, gy)

    def add(self, event, global_step):
        if not isinstance(event, dict):
            return

        reason = str(event.get("reason", ""))
        if reason not in ("offroad", "collision", "stuck", "overspeed"):
            return

        key = self._key(event)
        rec = self.cells.setdefault(
            key,
            {
                "count": 0,
                "nominal_count": 0,
                "dr_count": 0,
                "sum_x": 0.0,
                "sum_y": 0.0,
                "first_step": int(global_step),
                "last_step": int(global_step),
                "spawns": {},
                "reasons": {},
                "families": {},
            },
        )

        rec["count"] += 1
        rec["sum_x"] += float(event.get("world_x", 0.0))
        rec["sum_y"] += float(event.get("world_y", 0.0))
        rec["last_step"] = int(global_step)

        if bool(event.get("dr_active", False)):
            rec["dr_count"] += 1
        else:
            rec["nominal_count"] += 1

        spawn = str(int(event.get("spawn_number", 0)))
        rec["spawns"][spawn] = int(rec["spawns"].get(spawn, 0)) + 1

        rec["reasons"][reason] = int(rec["reasons"].get(reason, 0)) + 1

        families = event.get("dr_families", ())
        if isinstance(families, (list, tuple)):
            for family in families:
                family = str(family)
                rec["families"][family] = int(
                    rec["families"].get(family, 0)
                ) + 1

    def top(self, limit=8):
        rows = []
        for rec in self.cells.values():
            n = max(int(rec.get("count", 0)), 1)
            row = dict(rec)
            row["mean_x"] = float(rec.get("sum_x", 0.0)) / n
            row["mean_y"] = float(rec.get("sum_y", 0.0)) / n

            # Clean failures are treated as the most important evidence.
            if int(rec.get("nominal_count", 0)) >= 2:
                row["status"] = "PERSISTENT_CLEAN"
            elif int(rec.get("dr_count", 0)) >= 3 and int(rec.get("nominal_count", 0)) == 0:
                row["status"] = "DR_SENSITIVE"
            elif int(rec.get("count", 0)) >= 3:
                row["status"] = "REPEATED"
            else:
                row["status"] = "WATCH"

            rows.append(row)

        rows.sort(
            key=lambda x: (
                int(x.get("nominal_count", 0)),
                int(x.get("count", 0)),
            ),
            reverse=True,
        )
        return rows[:int(limit)]

    def to_dict(self):
        return {
            "grid_size": float(self.grid_size),
            "cells": self.cells,
        }

    @classmethod
    def load(cls):
        path = failure_cluster_path()
        obj = cls()
        if not os.path.isfile(path):
            return obj
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            obj.grid_size = float(data.get("grid_size", obj.grid_size))
            if isinstance(data.get("cells"), dict):
                obj.cells = data["cells"]
        except Exception:
            pass
        return obj

    def save(self):
        with open(
            failure_cluster_path(),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
                sort_keys=True,
            )

    def print_report(self, limit=8):
        rows = self.top(limit=limit)
        if not rows:
            print("  FAILURE ZONES | none yet")
            return

        print("  FAILURE ZONES | diagnostic only; NEVER auto-oversampled")
        for row in rows:
            print(
                "    {:16s} count={:3d} NOM/DR={:2d}/{:2d} "
                "mean=({:+.2f},{:+.2f}) age={:>7d}step "
                "spawns={} families={}".format(
                    str(row.get("status", "WATCH")),
                    int(row.get("count", 0)),
                    int(row.get("nominal_count", 0)),
                    int(row.get("dr_count", 0)),
                    float(row.get("mean_x", 0.0)),
                    float(row.get("mean_y", 0.0)),
                    int(row.get("last_step", 0)) - int(row.get("first_step", 0)),
                    row.get("spawns", {}),
                    row.get("families", {}),
                )
            )


def write_training_state(
    agent,
    global_step,
    episode,
    checkpoint_path,
    args,
):
    data = {
        "version": TRAINER_VERSION,
        "model_name": MODEL_NAME,
        "global_step": int(
            global_step
        ),
        "episode": int(
            episode
        ),
        "total_timesteps": int(
            args.total_timesteps
        ),
        "desired_speed_mps": float(
            args.desired_speed
        ),
        "action_std": float(
            agent.action_std
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "workers": int(
            args.workers
        ),
        "rollout_total": int(
            args.rollout_total
        ),
        "safe_spawns": [int(x) for x in args.safe_spawns],
        "recovery_min_episode_step": int(args.recovery_min_episode_step),
        "recovery_stable_ticks": int(args.recovery_stable_ticks),
        "recovery_max_events": int(args.recovery_max_events),
        "saved_at": (
            datetime.now()
            .isoformat(
                timespec="seconds"
            )
        ),
    }

    with open(
        state_path(),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            sort_keys=True,
        )

    return state_path()


def read_training_state():
    path = state_path()

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Missing V5 state: {}".format(
                path
            )
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if data.get("version") not in COMPATIBLE_STATE_VERSIONS:
        raise RuntimeError(
            "Training state is not compatible with this STABLE035 probe: {}".format(
                data.get("version")
            )
        )

    if data.get("model_name") != MODEL_NAME:
        raise RuntimeError(
            "Training state model_name mismatch."
        )

    return data


def prepare_fresh_or_resume(
    agent,
    args,
):
    files = policy_files()
    state_exists = os.path.isfile(
        state_path()
    )

    if not bool(args.resume):
        if files or state_exists:
            raise RuntimeError(
                "FRESH SAFETY STOP: STABLE035 namespace already contains "
                "checkpoint/state in '{}'. "
                "Use --resume true only if you intentionally want to continue "
                "this V5 run; otherwise archive that directory first.".format(
                    checkpoint_dir()
                )
            )

        init_checkpoint = str(args.init_checkpoint).strip()

        if init_checkpoint:
            if not os.path.isfile(init_checkpoint):
                raise FileNotFoundError(
                    "Init checkpoint missing: {}".format(init_checkpoint)
                )

            loaded = agent.load(
                checkpoint_path=init_checkpoint
            )
            agent.memory.clear()

            print(
                "NEW SINGLE-MAP STABLE RUN | initialized from {} | global_step=0".format(
                    loaded
                )
            )
            return (0, 0, loaded)

        print(
            "FRESH STABLE035 START | no checkpoint/state | global_step=0"
        )

        return (
            0,
            0,
            None,
        )

    # Resume path is intentionally strict.
    state = read_training_state()

    checkpoint_path = str(
        state.get(
            "checkpoint_path",
            "",
        )
    )

    if not checkpoint_path:
        raise RuntimeError(
            "V5 training state has no checkpoint_path."
        )

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            "Resume checkpoint missing: {}".format(
                checkpoint_path
            )
        )

    loaded = agent.load(
        checkpoint_path=checkpoint_path
    )

    agent.memory.clear()

    global_step = int(
        state.get(
            "global_step",
            0,
        )
    )

    episode = int(
        state.get(
            "episode",
            0,
        )
    )

    if global_step < 0:
        raise RuntimeError(
            "Invalid saved global_step."
        )

    print(
        "RESUME STABLE035 | checkpoint={} | step={} | episode={}".format(
            loaded,
            global_step,
            episode,
        )
    )

    return (
        global_step,
        episode,
        loaded,
    )


def make_writer(args):
    if (
        not bool(args.tensorboard)
        or SummaryWriter is None
    ):
        if (
            bool(args.tensorboard)
            and SummaryWriter is None
        ):
            print(
                "WARNING: TensorBoard unavailable."
            )

        return None

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    mode = (
        "resume"
        if args.resume
        else "fresh"
    )

    run_dir = os.path.join(
        "runs",
        "{}_{}_{}".format(
            MODEL_NAME,
            mode,
            stamp,
        ),
    )

    writer = SummaryWriter(
        run_dir
    )

    writer.add_text(
        "config",
        "\n".join(
            "{} = {}".format(
                key,
                value,
            )
            for key, value
            in sorted(
                vars(args).items()
            )
        ),
    )

    # TensorBoard dashboard groups. All x-axes are global_step.
    try:
        writer.add_custom_scalars({
            "01_BEHAVIOR": {
                "Survival": [
                    "Multiline",
                    ["behavior/episode_steps_mean", "behavior/progress_m_mean"],
                ],
                "Lateral error": [
                    "Multiline",
                    ["behavior/avg_abs_lateral_m", "behavior/max_abs_lateral_m"],
                ],
                "Termination": [
                    "Multiline",
                    ["behavior/offroad_rate", "behavior/collision_rate",
                     "behavior/stuck_rate", "behavior/time_limit_rate"],
                ],
            },
            "02_CONTROL": {
                "Speed": [
                    "Multiline",
                    ["control/avg_speed_mps", "control/avg_speed_cmd_mps"],
                ],
            },
            "03_CURRICULUM": {
                "Recovery": [
                    "Multiline",
                    ["curriculum/injected_events",
                     "curriculum/injected_success_rate",
                     "curriculum/allowed_fraction"],
                ],
            },
        })
    except Exception:
        pass

    print(
        "TensorBoard:",
        run_dir,
    )

    return writer


def state_dict_to_numpy(module):
    result = {}

    for key, value in (
        module.state_dict().items()
    ):
        result[key] = (
            value.detach()
            .cpu()
            .numpy()
            .copy()
        )

    return result


def load_numpy_state(
    module,
    state_numpy,
):
    current = module.state_dict()
    converted = {}

    for key, value in (
        state_numpy.items()
    ):
        if key not in current:
            raise RuntimeError(
                "Policy snapshot unexpected key: {}".format(
                    key
                )
            )

        converted[key] = (
            torch.as_tensor(
                value
            )
        )

    module.load_state_dict(
        converted,
        strict=True,
    )


def export_memory(memory):
    n = len(
        memory.rewards
    )

    if n <= 0:
        raise RuntimeError(
            "Cannot export empty rollout."
        )

    observations = (
        torch.stack(
            memory.observation,
            dim=0,
        )
        .cpu()
        .numpy()
    )

    raw_actions = (
        torch.stack(
            memory.raw_actions,
            dim=0,
        )
        .cpu()
        .numpy()
    )

    log_probs = (
        torch.stack(
            memory.log_probs,
            dim=0,
        )
        .reshape(-1)
        .cpu()
        .numpy()
    )

    return {
        "observations": (
            observations.astype(
                np.float32,
                copy=False,
            )
        ),
        "raw_actions": (
            raw_actions.astype(
                np.float32,
                copy=False,
            )
        ),
        "log_probs": (
            log_probs.astype(
                np.float32,
                copy=False,
            )
        ),
        "rewards": np.asarray(
            memory.rewards,
            dtype=np.float32,
        ),
        "terminateds": np.asarray(
            memory.terminateds,
            dtype=np.bool_,
        ),
        "truncateds": np.asarray(
            memory.truncateds,
            dtype=np.bool_,
        ),
        "bootstrap_values": np.asarray(
            memory.bootstrap_values,
            dtype=np.float32,
        ),
        "dones": np.asarray(
            memory.dones,
            dtype=np.bool_,
        ),
    }


def append_rollout_to_master(
    agent,
    rollout,
):
    keys = (
        "observations",
        "raw_actions",
        "log_probs",
        "rewards",
        "terminateds",
        "truncateds",
        "bootstrap_values",
        "dones",
    )

    lengths = {
        key: len(
            rollout[key]
        )
        for key in keys
    }

    if len(
        set(
            lengths.values()
        )
    ) != 1:
        raise RuntimeError(
            "Worker rollout length mismatch: {}".format(
                lengths
            )
        )

    n = int(
        lengths["rewards"]
    )

    if n <= 0:
        raise RuntimeError(
            "Worker rollout is empty."
        )

    # Mandatory return boundary between workers.
    if not bool(
        rollout["terminateds"][-1]
        or rollout["truncateds"][-1]
    ):
        raise RuntimeError(
            "Worker rollout has no final return boundary."
        )

    for index in range(n):
        agent.memory.observation.append(
            torch.as_tensor(
                rollout[
                    "observations"
                ][index],
                dtype=torch.float32,
            )
        )

        agent.memory.raw_actions.append(
            torch.as_tensor(
                rollout[
                    "raw_actions"
                ][index],
                dtype=torch.float32,
            )
        )

        agent.memory.log_probs.append(
            torch.as_tensor(
                rollout[
                    "log_probs"
                ][index],
                dtype=torch.float32,
            )
        )

        agent.memory.rewards.append(
            float(
                rollout[
                    "rewards"
                ][index]
            )
        )

        agent.memory.terminateds.append(
            bool(
                rollout[
                    "terminateds"
                ][index]
            )
        )

        agent.memory.truncateds.append(
            bool(
                rollout[
                    "truncateds"
                ][index]
            )
        )

        agent.memory.bootstrap_values.append(
            float(
                rollout[
                    "bootstrap_values"
                ][index]
            )
        )

        agent.memory.dones.append(
            bool(
                rollout[
                    "dones"
                ][index]
            )
        )

    return n


class EpisodeAccumulator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.reward = 0.0
        self.steps = 0
        self.progress_m = 0.0

        self.speed_sum = 0.0
        self.speed_max = 0.0
        self.speed_cmd_sum = 0.0
        self.abs_steer_sum = 0.0

        self.abs_lat_sum = 0.0
        self.abs_lat_max = 0.0
        self.abs_heading_sum = 0.0
        self.abs_heading_max = 0.0

        self.steer_gt_050 = 0
        self.steer_gt_075 = 0

        # Reward-manager recovery state diagnostics.
        self.recovery_attempts = 0
        self.recovery_successes = 0

        # Actual curriculum injections.
        self.disturbance_attempts = 0
        self.disturbance_successes = 0

    def add(
        self,
        action,
        reward,
        info,
    ):
        self.reward += float(
            reward
        )
        self.steps += 1

        speed = float(
            info.get(
                "speed_mps",
                0.0,
            )
        )

        speed_cmd = float(
            info.get(
                "speed_cmd_mps",
                float(action[1]),
            )
        )

        lat = abs(
            float(
                info.get(
                    "lateral_error_m",
                    0.0,
                )
            )
        )

        heading = abs(
            float(
                info.get(
                    "heading_error_rad",
                    0.0,
                )
            )
        )

        self.speed_sum += speed
        self.speed_max = max(
            self.speed_max,
            speed,
        )

        self.speed_cmd_sum += speed_cmd

        control = info.get("control", {})
        steer_value = float(
            control.get(
                "actual_steer_cmd",
                float(action[0]),
            )
        ) if isinstance(control, dict) else float(action[0])

        abs_steer = abs(steer_value)
        self.abs_steer_sum += abs_steer
        if abs_steer >= 0.50:
            self.steer_gt_050 += 1
        if abs_steer >= 0.75:
            self.steer_gt_075 += 1

        self.abs_lat_sum += lat
        self.abs_lat_max = max(
            self.abs_lat_max,
            lat,
        )
        self.abs_heading_sum += heading
        self.abs_heading_max = max(
            self.abs_heading_max,
            heading,
        )

        self.progress_m += float(
            info.get(
                "forward_progress_m",
                0.0,
            )
        )

        if bool(
            info.get(
                "recovery_trigger_event",
                False,
            )
        ):
            self.recovery_attempts += 1

        if bool(
            info.get(
                "recovery_success_event",
                False,
            )
        ):
            self.recovery_successes += 1


        if bool(info.get("disturbance_trigger_event", False)):
            self.disturbance_attempts += 1

        if bool(info.get("disturbance_success_event", False)):
            self.disturbance_successes += 1

    def finish(self, info):
        n = max(
            int(self.steps),
            1,
        )

        summary = {
            "reward": float(
                self.reward
            ),
            "steps": int(
                self.steps
            ),
            "avg_speed_mps": (
                self.speed_sum / n
            ),
            "max_speed_mps": float(
                self.speed_max
            ),
            "avg_speed_cmd_mps": (
                self.speed_cmd_sum / n
            ),
            "avg_abs_steer": (
                self.abs_steer_sum / n
            ),
            "steer_gt_050_rate": (
                float(self.steer_gt_050) / float(n)
            ),
            "steer_gt_075_rate": (
                float(self.steer_gt_075) / float(n)
            ),
            "avg_abs_lateral_m": (
                self.abs_lat_sum / n
            ),
            "max_abs_lateral_m": float(
                self.abs_lat_max
            ),
            "avg_abs_heading_rad": (
                self.abs_heading_sum / n
            ),
            "max_abs_heading_rad": float(
                self.abs_heading_max
            ),
            "progress_m": float(
                self.progress_m
            ),
            "reason": info.get(
                "termination_reason"
            ),
            "terminated": bool(
                info.get(
                    "terminated",
                    False,
                )
            ),
            "truncated": bool(
                info.get(
                    "truncated",
                    False,
                )
            ),
            "recovery_attempts": int(
                self.recovery_attempts
            ),
            "recovery_successes": int(
                self.recovery_successes
            ),
            "disturbance_attempts": int(
                self.disturbance_attempts
            ),
            "disturbance_successes": int(
                self.disturbance_successes
            ),
        }

        self.reset()

        return summary


class RolloutDiagnostics(object):
    """
    Per-step diagnostics independent of episode boundaries.
    This is what lets us verify speed learning by recovery state every update.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = 0
        self.final_reward_sum = 0.0

        self.state = {}

        for name in STATE_NAMES:
            self.state[name] = {
                "count": 0,
                "target_sum": 0.0,
                "cmd_sum": 0.0,
                "speed_sum": 0.0,
                "abs_cmd_error_sum": 0.0,
                "abs_speed_error_sum": 0.0,
                "reward_sum": 0.0,
            }

        self.reward_term_sums = {
            name: 0.0
            for name in REWARD_TERM_NAMES
        }

        self.recovery_trigger_events = 0
        self.recovery_success_events = 0

        self.abs_lat_sum = 0.0
        self.abs_lat_max = 0.0
        self.abs_heading_sum = 0.0
        self.abs_heading_max = 0.0
        self.abs_preview_heading_sum = 0.0
        self.abs_preview_heading_max = 0.0
        self.preview_lookahead_distance_sum = 0.0
        self.preview_lookahead_distance_count = 0
        self.preview_fallback_count = 0
        self.progress_sum = 0.0
        self.speed_sum = 0.0
        self.speed_cmd_sum = 0.0
        self.abs_steer_sum = 0.0
        self.steer_gt_050 = 0
        self.steer_gt_075 = 0

        # Steering stability diagnostics.
        self.prev_actual_steer = None
        self.steer_delta_abs_sum = 0.0
        self.steer_delta_abs_max = 0.0
        self.steer_delta_count = 0
        self.steer_sign_flip_count = 0

        self.prev_raw_steer = None
        self.abs_raw_steer_sum = 0.0
        self.raw_steer_delta_abs_sum = 0.0
        self.raw_steer_delta_abs_max = 0.0
        self.raw_steer_delta_count = 0
        self.raw_steer_sign_flip_count = 0
        self.steer_rate_limiter_hit_count = 0

        # Privileged oracle-teacher reward diagnostics.
        self.teacher_abs_target_sum = 0.0
        self.teacher_abs_error_sum = 0.0
        self.teacher_factor_sum = 0.0
        self.teacher_strong_count = 0
        self.teacher_strong_abs_error_sum = 0.0

        self.camera_lag_sum = 0.0
        self.camera_lag_count = 0
        self.camera_lag_max = 0
        self.curriculum_allowed_samples = 0
        self.disturbance_trigger_events = 0
        self.disturbance_success_events = 0

    def reset_steer_history(self):
        # Episode reset must not be counted as a physical steering jump.
        self.prev_actual_steer = None
        self.prev_raw_steer = None

    def add(
        self,
        reward,
        info,
    ):
        self.samples += 1
        self.final_reward_sum += float(
            reward
        )

        state_name = str(
            info.get(
                "recovery_state",
                "unknown",
            )
        )

        target = float(
            info.get(
                "adaptive_target_speed_mps",
                0.0,
            )
        )

        cmd = float(
            info.get(
                "speed_cmd_mps",
                0.0,
            )
        )

        speed = float(
            info.get(
                "speed_mps",
                0.0,
            )
        )

        if state_name in self.state:
            state = self.state[
                state_name
            ]

            state["count"] += 1
            state["target_sum"] += target
            state["cmd_sum"] += cmd
            state["speed_sum"] += speed
            state[
                "abs_cmd_error_sum"
            ] += abs(
                cmd - target
            )
            state[
                "abs_speed_error_sum"
            ] += abs(
                speed - target
            )
            state[
                "reward_sum"
            ] += float(
                reward
            )

        terms = info.get(
            "reward_terms",
            {},
        )

        if isinstance(
            terms,
            dict,
        ):
            for name in (
                REWARD_TERM_NAMES
            ):
                self.reward_term_sums[
                    name
                ] += float(
                    terms.get(
                        name,
                        0.0,
                    )
                )

        lat = abs(float(info.get("lateral_error_m", 0.0)))
        heading = abs(float(info.get("heading_error_rad", 0.0)))
        preview_heading = abs(
            float(
                info.get(
                    "preview_heading_error_rad",
                    info.get(
                        "heading_error_rad",
                        0.0,
                    ),
                )
            )
        )
        preview_distance = float(
            info.get(
                "preview_lookahead_distance_m",
                0.0,
            )
        )

        self.abs_lat_sum += lat
        self.abs_lat_max = max(self.abs_lat_max, lat)
        self.abs_heading_sum += heading
        self.abs_heading_max = max(self.abs_heading_max, heading)
        self.abs_preview_heading_sum += preview_heading
        self.abs_preview_heading_max = max(
            self.abs_preview_heading_max,
            preview_heading,
        )
        if preview_distance > 0.0:
            self.preview_lookahead_distance_sum += preview_distance
            self.preview_lookahead_distance_count += 1
        else:
            self.preview_fallback_count += 1
        self.progress_sum += float(info.get("forward_progress_m", 0.0))
        self.speed_sum += float(info.get("speed_mps", 0.0))
        self.speed_cmd_sum += float(info.get("speed_cmd_mps", 0.0))

        control = info.get("control", {})
        if isinstance(control, dict):
            steer = float(control.get("actual_steer_cmd", 0.0))
        else:
            steer = 0.0
        abs_steer = abs(steer)
        self.abs_steer_sum += abs_steer
        if abs_steer >= 0.50:
            self.steer_gt_050 += 1
        if abs_steer >= 0.75:
            self.steer_gt_075 += 1

        raw_steer = 0.0
        limiter_hit = False
        if isinstance(control, dict):
            raw_steer = float(
                control.get(
                    "raw_policy_steer",
                    steer,
                )
            )
            limiter_hit = bool(
                control.get(
                    "steer_rate_limiter_hit",
                    False,
                )
            )

        self.abs_raw_steer_sum += abs(
            float(raw_steer)
        )
        if limiter_hit:
            self.steer_rate_limiter_hit_count += 1

        if self.prev_raw_steer is not None:
            raw_delta = abs(
                float(raw_steer)
                - float(self.prev_raw_steer)
            )
            self.raw_steer_delta_abs_sum += float(
                raw_delta
            )
            self.raw_steer_delta_abs_max = max(
                float(
                    self.raw_steer_delta_abs_max
                ),
                float(raw_delta),
            )
            self.raw_steer_delta_count += 1

            if (
                abs(float(raw_steer)) >= 0.12
                and
                abs(float(self.prev_raw_steer)) >= 0.12
                and
                float(raw_steer)
                * float(self.prev_raw_steer)
                < 0.0
            ):
                self.raw_steer_sign_flip_count += 1

        self.prev_raw_steer = float(
            raw_steer
        )

        teacher_target = float(
            info.get(
                "teacher_steer_target_unit",
                info.get(
                    "teacher_steer_target_preaction",
                    0.0,
                ),
            )
        )
        teacher_error = abs(
            float(
                steer
            )
            - float(
                teacher_target
            )
        )
        teacher_factor = float(
            info.get(
                "teacher_steer_factor",
                0.0,
            )
        )

        self.teacher_abs_target_sum += abs(
            teacher_target
        )
        self.teacher_abs_error_sum += float(
            teacher_error
        )
        self.teacher_factor_sum += float(
            teacher_factor
        )

        if abs(teacher_target) >= 0.15:
            self.teacher_strong_count += 1
            self.teacher_strong_abs_error_sum += float(
                teacher_error
            )

        if self.prev_actual_steer is not None:
            steer_delta = abs(
                float(steer)
                - float(self.prev_actual_steer)
            )
            self.steer_delta_abs_sum += float(steer_delta)
            self.steer_delta_abs_max = max(
                float(self.steer_delta_abs_max),
                float(steer_delta),
            )
            self.steer_delta_count += 1

            # Count only meaningful left/right reversals; tiny zero chatter does
            # not count as an oscillation event.
            if (
                abs(float(steer)) >= 0.12
                and abs(float(self.prev_actual_steer)) >= 0.12
                and float(steer) * float(self.prev_actual_steer) < 0.0
            ):
                self.steer_sign_flip_count += 1

        self.prev_actual_steer = float(steer)

        camera_lag = info.get("camera_frame_lag")
        if camera_lag is not None:
            camera_lag = int(camera_lag)
            self.camera_lag_sum += float(camera_lag)
            self.camera_lag_count += 1
            self.camera_lag_max = max(self.camera_lag_max, camera_lag)

        if bool(info.get("recovery_curriculum_allowed", False)):
            self.curriculum_allowed_samples += 1

        if bool(info.get("disturbance_trigger_event", False)):
            self.disturbance_trigger_events += 1

        if bool(info.get("disturbance_success_event", False)):
            self.disturbance_success_events += 1

        if bool(
            info.get(
                "recovery_trigger_event",
                False,
            )
        ):
            self.recovery_trigger_events += 1

        if bool(
            info.get(
                "recovery_success_event",
                False,
            )
        ):
            self.recovery_success_events += 1

    def export(self):
        return {
            "samples": int(
                self.samples
            ),
            "final_reward_sum": float(
                self.final_reward_sum
            ),
            "state": self.state,
            "reward_term_sums": (
                self.reward_term_sums
            ),
            "recovery_trigger_events": int(
                self.recovery_trigger_events
            ),
            "recovery_success_events": int(
                self.recovery_success_events
            ),
            "abs_lat_sum": float(self.abs_lat_sum),
            "abs_lat_max": float(self.abs_lat_max),
            "abs_heading_sum": float(self.abs_heading_sum),
            "abs_heading_max": float(self.abs_heading_max),
            "abs_preview_heading_sum": float(
                self.abs_preview_heading_sum
            ),
            "abs_preview_heading_max": float(
                self.abs_preview_heading_max
            ),
            "preview_lookahead_distance_sum": float(
                self.preview_lookahead_distance_sum
            ),
            "preview_lookahead_distance_count": int(
                self.preview_lookahead_distance_count
            ),
            "preview_fallback_count": int(
                self.preview_fallback_count
            ),
            "progress_sum": float(self.progress_sum),
            "speed_sum": float(self.speed_sum),
            "speed_cmd_sum": float(self.speed_cmd_sum),
            "abs_steer_sum": float(self.abs_steer_sum),
            "steer_gt_050": int(self.steer_gt_050),
            "steer_gt_075": int(self.steer_gt_075),
            "steer_delta_abs_sum": float(self.steer_delta_abs_sum),
            "steer_delta_abs_max": float(self.steer_delta_abs_max),
            "steer_delta_count": int(self.steer_delta_count),
            "steer_sign_flip_count": int(self.steer_sign_flip_count),
            "abs_raw_steer_sum": float(
                self.abs_raw_steer_sum
            ),
            "raw_steer_delta_abs_sum": float(
                self.raw_steer_delta_abs_sum
            ),
            "raw_steer_delta_abs_max": float(
                self.raw_steer_delta_abs_max
            ),
            "raw_steer_delta_count": int(
                self.raw_steer_delta_count
            ),
            "raw_steer_sign_flip_count": int(
                self.raw_steer_sign_flip_count
            ),
            "steer_rate_limiter_hit_count": int(
                self.steer_rate_limiter_hit_count
            ),
            "teacher_abs_target_sum": float(
                self.teacher_abs_target_sum
            ),
            "teacher_abs_error_sum": float(
                self.teacher_abs_error_sum
            ),
            "teacher_factor_sum": float(
                self.teacher_factor_sum
            ),
            "teacher_strong_count": int(
                self.teacher_strong_count
            ),
            "teacher_strong_abs_error_sum": float(
                self.teacher_strong_abs_error_sum
            ),
            "camera_lag_sum": float(self.camera_lag_sum),
            "camera_lag_count": int(self.camera_lag_count),
            "camera_lag_max": int(self.camera_lag_max),
            "curriculum_allowed_samples": int(self.curriculum_allowed_samples),
            "disturbance_trigger_events": int(self.disturbance_trigger_events),
            "disturbance_success_events": int(self.disturbance_success_events),
        }


def connect_worker(
    host,
    port,
    timeout,
    expected_map,
):
    client = carla.Client(
        str(host),
        int(port),
    )

    client.set_timeout(
        float(timeout)
    )

    world = client.get_world()

    map_name = str(
        world.get_map().name
    )

    expected = (
        str(expected_map)
        .replace("\\", "/")
        .lower()
    )

    actual = (
        map_name
        .replace("\\", "/")
        .lower()
    )

    if (
        expected
        and expected not in actual
    ):
        raise RuntimeError(
            "CARLA port {} map mismatch: actual='{}', "
            "expected contains '{}'".format(
                port,
                map_name,
                expected_map,
            )
        )

    # Keep the connection baseline deterministic before per-episode vision DR.
    world.set_weather(
        carla.WeatherParameters.CloudyNoon
    )

    return (
        client,
        world,
        map_name,
    )


def current_reward_profile(env):
    reward_fn = env.reward_manager.reward_fn
    min_speed, target_speed, max_speed = reward_fn.speed_limits(
        env.desired_speed_mps
    )
    return {
        "min": float(min_speed),
        "target": float(target_speed),
        "max": float(max_speed),
    }


def verify_reward_profile(profile):
    maximum = float(profile.get("max", -1.0))
    target = float(profile.get("target", -1.0))
    minimum = float(profile.get("min", -1.0))

    if maximum <= 0.0:
        raise RuntimeError("Invalid STABLE035 speed profile: {}".format(profile))

    expected_target = maximum * float(EXPECTED_SPEED_PROFILE["target_ratio"])
    expected_min = maximum * float(EXPECTED_SPEED_PROFILE["min_ratio"])

    if abs(target - expected_target) > 1e-6:
        raise RuntimeError(
            "TARGET speed mismatch actual={} expected={}".format(
                target, expected_target
            )
        )
    if abs(minimum - expected_min) > 1e-6:
        raise RuntimeError(
            "MIN speed mismatch actual={} expected={}".format(
                minimum, expected_min
            )
        )


def worker_main(
    worker_id,
    conn,
    cfg,
):
    env = None

    try:
        torch.set_num_threads(
            int(
                cfg[
                    "worker_torch_threads"
                ]
            )
        )

        seed = int(
            cfg["seed"]
        )

        seed_everything(
            seed
        )

        configure_recovery_scenarios_v5(
            # Spawn perturbation is hard-disabled in ONE-SPAWN mode.
            recovery_spawn_probability=0.0,
            spawn_lateral_min_m=0.0,
            spawn_lateral_max_m=0.0,
            spawn_heading_min_deg=0.0,
            spawn_heading_max_deg=0.0,
            disturbance_probability=float(cfg["disturbance_prob"]),
            disturbance_duration_min_s=float(cfg["disturbance_duration_min_s"]),
            disturbance_duration_max_s=float(cfg["disturbance_duration_max_s"]),
            disturbance_steer_min=float(cfg["disturbance_steer_min"]),
            disturbance_steer_max=float(cfg["disturbance_steer_max"]),
            recovery_min_episode_step=int(cfg["recovery_min_episode_step"]),
            recovery_stable_ticks=int(cfg["recovery_stable_ticks"]),
            recovery_stable_lateral_m=float(cfg["recovery_stable_lateral_m"]),
            recovery_stable_heading_deg=float(cfg["recovery_stable_heading_deg"]),
            recovery_stable_speed_mps=float(cfg["recovery_stable_speed_mps"]),
            recovery_cooldown_ticks=int(cfg["recovery_cooldown_ticks"]),
            recovery_max_events_per_episode=int(cfg["recovery_max_events"]),
            seed=seed,
        )

        (
            client,
            world,
            map_name,
        ) = connect_worker(
            host=cfg["host"],
            port=cfg["port"],
            timeout=cfg[
                "carla_timeout"
            ],
            expected_map=cfg[
                "expected_map"
            ],
        )

        # Independent deterministic stream for dynamics/vision sensor DR.
        dr_seed = (
            int(seed)
            + 60000
        )

        env = CarlaEnvironmentRGBV5(
            client=client,
            world=world,
            town=(
                "worker_{}".format(
                    worker_id
                )
            ),
            safe_spawn_numbers=(
                cfg[
                    "safe_spawns"
                ]
            ),
            desired_speed_mps=float(
                cfg[
                    "desired_speed"
                ]
            ),
            policy_speed_max_mps=float(
                cfg[
                    "desired_speed"
                ]
            ),
            steer_rate_limit_per_s=float(
                cfg["steer_rate_limit"]
            ),
            speed_rate_limit_mps2=float(
                cfg["speed_rate_limit"]
            ),
            steer_deadband=float(
                cfg["steer_deadband"]
            ),
            dr_family_budget=int(
                cfg["dr_family_budget"]
            ),
            max_episode_seconds=float(
                cfg[
                    "max_episode_seconds"
                ]
            ),
            encoder_device=cfg[
                "encoder_device"
            ],
            dynamics_dr_enabled=False,
            vision_dr_enabled=False,
            sensor_dr_enabled=False,
            domain_randomization_seed=(
                dr_seed
            ),
            dr_episode_probability=0.0,
            recovery_scenarios_enabled=False,
        )

        local_agent = PPOAgent(
            town=(
                "__v5_clean_worker_{}".format(
                    worker_id
                )
            ),
            action_std_init=float(
                cfg[
                    "action_std_init"
                ]
            ),
            device=cfg[
                "worker_ppo_device"
            ],
        )

        observation = env.reset()

        profile = current_reward_profile(
            env
        )
        verify_reward_profile(
            profile
        )

        episode_stats = (
            EpisodeAccumulator()
        )
        completed_episode_count = 0
        current_phase_signature = None

        conn.send(
            {
                "type": "ready",
                "worker": int(
                    worker_id
                ),
                "port": int(
                    cfg["port"]
                ),
                "seed": int(
                    seed
                ),
                "map": str(
                    map_name
                ),
                "control_hz": float(
                    env.control_hz
                ),
                "dr_seed": int(
                    dr_seed
                ),
                "reward_profile": (
                    profile
                ),
            }
        )

        while True:
            command = conn.recv()
            kind = command.get(
                "cmd"
            )

            if kind == "close":
                break

            if kind != "collect":
                raise RuntimeError(
                    "Unknown worker command: {}".format(
                        kind
                    )
                )

            # Exact shared on-policy snapshot.
            load_numpy_state(
                local_agent.policy,
                command[
                    "policy_state"
                ],
            )

            load_numpy_state(
                local_agent.old_policy,
                command[
                    "policy_state"
                ],
            )

            local_agent.set_action_std(
                float(
                    command[
                        "action_std"
                    ]
                )
            )

            local_agent.memory.clear()

            env.set_desired_speed_mps(
                float(
                    command[
                        "desired_speed"
                    ]
                )
            )

            phase_signature = (
                str(command.get("phase_name", "CLEAN")),
                round(float(command.get("dr_probability", 0.0)), 4),
                bool(command.get("dr_dynamics", False)),
                bool(command.get("dr_vision", False)),
                bool(command.get("dr_sensor", False)),
                int(command.get("dr_family_budget", 1)),
            )

            env.set_stable_dr_control(
                probability=float(command.get("dr_probability", 0.0)),
                dynamics_enabled=bool(command.get("dr_dynamics", False)),
                vision_enabled=bool(command.get("dr_vision", False)),
                sensor_enabled=bool(command.get("dr_sensor", False)),
                family_budget=int(command.get("dr_family_budget", 1)),
            )

            if phase_signature != current_phase_signature:
                observation = env.reset()
                episode_stats = EpisodeAccumulator()
                current_phase_signature = phase_signature

            requested_steps = int(
                command["steps"]
            )

            completed_episodes = []
            failure_events = []
            diagnostics = (
                RolloutDiagnostics()
            )

            rollout_start = (
                time.time()
            )

            artificial_final_cut = False

            for local_index in range(
                requested_steps
            ):
                action = (
                    local_agent.get_action(
                        observation,
                        train=True,
                    )
                )

                (
                    next_observation,
                    reward,
                    done,
                    info,
                ) = env.step(
                    action
                )

                actual_terminated = bool(
                    info.get(
                        "terminated",
                        False,
                    )
                )

                actual_truncated = bool(
                    info.get(
                        "truncated",
                        False,
                    )
                )

                is_last_sample = bool(
                    local_index
                    == requested_steps - 1
                )

                artificial_final_cut = bool(
                    is_last_sample
                    and not actual_terminated
                    and not actual_truncated
                )

                train_truncated = bool(
                    actual_truncated
                    or artificial_final_cut
                )

                local_agent.record_outcome(
                    reward=reward,
                    terminated=(
                        actual_terminated
                    ),
                    truncated=(
                        train_truncated
                    ),
                    next_obs=(
                        next_observation
                        if train_truncated
                        else None
                    ),
                )

                episode_stats.add(
                    action=action,
                    reward=reward,
                    info=info,
                )

                diagnostics.add(
                    reward=reward,
                    info=info,
                )

                if done:
                    # Prevent the first steering sample of the next episode
                    # from being differenced against the last sample of this one.
                    diagnostics.reset_steer_history()

                    completed_episode_count += 1

                    completed_episodes.append(
                        episode_stats.finish(
                            info
                        )
                    )

                    reason = str(
                        info.get(
                            "termination_reason",
                            "",
                        )
                    )
                    if reason in (
                        "offroad",
                        "collision",
                        "stuck",
                        "overspeed",
                    ):
                        failure_events.append(
                            {
                                "reason": reason,
                                "world_x": float(info.get("world_x", 0.0)),
                                "world_y": float(info.get("world_y", 0.0)),
                                "spawn_number": int(info.get("spawn_number", 0) or 0),
                                "dr_active": bool(info.get("dr_episode_active", False)),
                                "dr_families": list(info.get("dr_episode_families", ())),
                            }
                        )

                    observation = (
                        env.reset()
                    )
                else:
                    observation = (
                        next_observation
                    )

            rollout = export_memory(
                local_agent.memory
            )

            local_agent.memory.clear()

            conn.send(
                {
                    "type": "rollout",
                    "worker": int(
                        worker_id
                    ),
                    "samples": int(
                        requested_steps
                    ),
                    "wall_s": float(
                        time.time()
                        - rollout_start
                    ),
                    "rollout": rollout,
                    "diagnostics": (
                        diagnostics.export()
                    ),
                    "episodes": (
                        completed_episodes
                    ),
                    "failure_events": failure_events,
                    "phase_name": str(command.get("phase_name", "CLEAN")),
                    "worker_episode_count": int(
                        completed_episode_count
                    ),
                    "artificial_final_cut": bool(
                        artificial_final_cut
                    ),
                }
            )

    except KeyboardInterrupt:
        pass

    except Exception as error:
        try:
            conn.send(
                {
                    "type": "error",
                    "worker": int(
                        worker_id
                    ),
                    "error": repr(
                        error
                    ),
                    "traceback": (
                        traceback.format_exc()
                    ),
                }
            )
        except Exception:
            pass

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        try:
            conn.close()
        except Exception:
            pass


def split_steps(
    total,
    worker_count,
):
    total = int(total)
    worker_count = int(
        worker_count
    )

    base = (
        total // worker_count
    )
    remainder = (
        total % worker_count
    )

    return [
        base
        + (
            1
            if index < remainder
            else 0
        )
        for index in range(
            worker_count
        )
    ]


def maybe_decay_action_std(
    agent,
    global_step,
    previous_bucket,
    args,
):
    bucket = int(
        int(global_step)
        // int(
            args.action_std_decay_freq
        )
    )

    if bucket <= previous_bucket:
        return previous_bucket

    for _ in range(
        bucket - previous_bucket
    ):
        if (
            agent.action_std
            <= float(
                args.action_std_min
            )
            + 1e-12
        ):
            break

        agent.decay_action_std(
            action_std_decay_rate=float(
                args.action_std_decay
            ),
            min_action_std=float(
                args.action_std_min
            ),
        )

    print(
        "ACTION STD | step={} | std={:.4f}".format(
            global_step,
            agent.action_std,
        )
    )

    return bucket


def save_checkpoint(
    agent,
    global_step,
    episode,
    args,
):
    path = agent.save()

    state = write_training_state(
        agent=agent,
        global_step=global_step,
        episode=episode,
        checkpoint_path=path,
        args=args,
    )

    print(
        "CHECKPOINT | step={} | policy={} | state={}".format(
            global_step,
            path,
            state,
        )
    )

    return path


def aggregate_diagnostics(
    worker_messages,
):
    result = {
        "samples": 0,
        "final_reward_sum": 0.0,
        "state": {},
        "reward_term_sums": {
            name: 0.0
            for name
            in REWARD_TERM_NAMES
        },
        "recovery_trigger_events": 0,
        "recovery_success_events": 0,
        "abs_lat_sum": 0.0,
        "abs_lat_max": 0.0,
        "abs_heading_sum": 0.0,
        "abs_heading_max": 0.0,
        "abs_preview_heading_sum": 0.0,
        "abs_preview_heading_max": 0.0,
        "preview_lookahead_distance_sum": 0.0,
        "preview_lookahead_distance_count": 0,
        "preview_fallback_count": 0,
        "progress_sum": 0.0,
        "speed_sum": 0.0,
        "speed_cmd_sum": 0.0,
        "abs_steer_sum": 0.0,
        "steer_gt_050": 0,
        "steer_gt_075": 0,
        "steer_delta_abs_sum": 0.0,
        "steer_delta_abs_max": 0.0,
        "steer_delta_count": 0,
        "steer_sign_flip_count": 0,
        "abs_raw_steer_sum": 0.0,
        "raw_steer_delta_abs_sum": 0.0,
        "raw_steer_delta_abs_max": 0.0,
        "raw_steer_delta_count": 0,
        "raw_steer_sign_flip_count": 0,
        "steer_rate_limiter_hit_count": 0,
        "teacher_abs_target_sum": 0.0,
        "teacher_abs_error_sum": 0.0,
        "teacher_factor_sum": 0.0,
        "teacher_strong_count": 0,
        "teacher_strong_abs_error_sum": 0.0,
        "camera_lag_sum": 0.0,
        "camera_lag_count": 0,
        "camera_lag_max": 0,
        "curriculum_allowed_samples": 0,
        "disturbance_trigger_events": 0,
        "disturbance_success_events": 0,
    }

    for state_name in STATE_NAMES:
        result["state"][
            state_name
        ] = {
            "count": 0,
            "target_sum": 0.0,
            "cmd_sum": 0.0,
            "speed_sum": 0.0,
            "abs_cmd_error_sum": 0.0,
            "abs_speed_error_sum": 0.0,
            "reward_sum": 0.0,
        }

    for message in worker_messages:
        diag = message[
            "diagnostics"
        ]

        result["samples"] += int(
            diag["samples"]
        )

        result[
            "final_reward_sum"
        ] += float(
            diag[
                "final_reward_sum"
            ]
        )

        result[
            "recovery_trigger_events"
        ] += int(
            diag[
                "recovery_trigger_events"
            ]
        )

        result[
            "recovery_success_events"
        ] += int(
            diag[
                "recovery_success_events"
            ]
        )

        for key in (
            "abs_lat_sum",
            "abs_heading_sum",
            "abs_preview_heading_sum",
            "preview_lookahead_distance_sum",
            "progress_sum",
            "speed_sum",
            "speed_cmd_sum",
            "abs_steer_sum",
            "steer_delta_abs_sum",
            "abs_raw_steer_sum",
            "raw_steer_delta_abs_sum",
            "teacher_abs_target_sum",
            "teacher_abs_error_sum",
            "teacher_factor_sum",
            "teacher_strong_abs_error_sum",
            "camera_lag_sum",
        ):
            result[key] += float(diag.get(key, 0.0))

        result["abs_lat_max"] = max(
            float(result["abs_lat_max"]),
            float(diag.get("abs_lat_max", 0.0)),
        )
        result["abs_heading_max"] = max(
            float(result["abs_heading_max"]),
            float(diag.get("abs_heading_max", 0.0)),
        )
        result["abs_preview_heading_max"] = max(
            float(result["abs_preview_heading_max"]),
            float(diag.get("abs_preview_heading_max", 0.0)),
        )
        result["steer_delta_abs_max"] = max(
            float(result["steer_delta_abs_max"]),
            float(diag.get("steer_delta_abs_max", 0.0)),
        )
        result["raw_steer_delta_abs_max"] = max(
            float(result["raw_steer_delta_abs_max"]),
            float(diag.get("raw_steer_delta_abs_max", 0.0)),
        )
        result["camera_lag_max"] = max(
            int(result["camera_lag_max"]),
            int(diag.get("camera_lag_max", 0)),
        )

        for key in (
            "steer_gt_050",
            "steer_gt_075",
            "steer_delta_count",
            "steer_sign_flip_count",
            "preview_lookahead_distance_count",
            "preview_fallback_count",
            "raw_steer_delta_count",
            "raw_steer_sign_flip_count",
            "steer_rate_limiter_hit_count",
            "teacher_strong_count",
            "camera_lag_count",
            "curriculum_allowed_samples",
            "disturbance_trigger_events",
            "disturbance_success_events",
        ):
            result[key] += int(diag.get(key, 0))

        for state_name in STATE_NAMES:
            source = diag[
                "state"
            ][state_name]

            target = result[
                "state"
            ][state_name]

            for key in (
                "count",
                "target_sum",
                "cmd_sum",
                "speed_sum",
                "abs_cmd_error_sum",
                "abs_speed_error_sum",
                "reward_sum",
            ):
                target[key] += source[
                    key
                ]

        for name in REWARD_TERM_NAMES:
            result[
                "reward_term_sums"
            ][name] += float(
                diag[
                    "reward_term_sums"
                ][name]
            )

    return result


def aggregate_episode_metrics(
    worker_messages,
):
    episodes = []

    for message in worker_messages:
        episodes.extend(message["episodes"])

    count = int(len(episodes))

    if count <= 0:
        return {
            "episodes": 0,
            "recovery_attempts": 0,
            "recovery_successes": 0,
            "recovery_rate": 0.0,
            "disturbance_attempts": 0,
            "disturbance_successes": 0,
            "disturbance_success_rate": 0.0,
        }

    def mean(key):
        return float(np.mean([
            float(item.get(key, 0.0))
            for item in episodes
        ]))

    attempts = sum(int(item.get("recovery_attempts", 0)) for item in episodes)
    successes = sum(int(item.get("recovery_successes", 0)) for item in episodes)
    dist_attempts = sum(int(item.get("disturbance_attempts", 0)) for item in episodes)
    dist_successes = sum(int(item.get("disturbance_successes", 0)) for item in episodes)

    reasons = [str(item.get("reason")) for item in episodes]

    return {
        "episodes": count,
        "recovery_attempts": int(attempts),
        "recovery_successes": int(successes),
        "recovery_rate": (
            float(successes) / float(attempts)
            if attempts > 0 else 0.0
        ),
        "disturbance_attempts": int(dist_attempts),
        "disturbance_successes": int(dist_successes),
        "disturbance_success_rate": (
            float(dist_successes) / float(dist_attempts)
            if dist_attempts > 0 else 0.0
        ),
        "avg_reward": mean("reward"),
        "avg_steps": mean("steps"),
        "avg_progress_m": mean("progress_m"),
        "avg_speed_mps": mean("avg_speed_mps"),
        "avg_speed_cmd_mps": mean("avg_speed_cmd_mps"),
        "avg_abs_steer": mean("avg_abs_steer"),
        "steer_gt_050_rate": mean("steer_gt_050_rate"),
        "steer_gt_075_rate": mean("steer_gt_075_rate"),
        "avg_abs_lateral_m": mean("avg_abs_lateral_m"),
        "max_abs_lateral_m": float(max(
            float(item.get("max_abs_lateral_m", 0.0))
            for item in episodes
        )),
        "avg_abs_heading_deg": float(np.degrees(
            mean("avg_abs_heading_rad")
        )),
        "max_abs_heading_deg": float(np.degrees(max(
            float(item.get("max_abs_heading_rad", 0.0))
            for item in episodes
        ))),
        "offroad_rate": float(sum(r == "offroad" for r in reasons)) / float(count),
        "collision_rate": float(sum(r == "collision" for r in reasons)) / float(count),
        "stuck_rate": float(sum(r == "stuck" for r in reasons)) / float(count),
        "time_limit_rate": float(sum(r == "time_limit" for r in reasons)) / float(count),
    }


def _safe_mean(
    total,
    count,
):
    if int(count) <= 0:
        return 0.0

    return (
        float(total)
        / float(count)
    )


def print_state_diagnostics(
    diagnostics,
):
    total_samples = max(
        int(diagnostics["samples"]),
        1,
    )
    n = float(total_samples)

    # Recovery-state labels are deliberately NOT printed here. They remain
    # internal diagnostics only and do not alter reward or speed.
    state_total_target = 0.0
    state_total_count = 0
    for state_name in STATE_NAMES:
        values = diagnostics["state"][state_name]
        state_total_target += float(values["target_sum"])
        state_total_count += int(values["count"])

    target_mean = (
        state_total_target / float(max(state_total_count, 1))
    )

    delta_count = max(
        int(diagnostics.get("steer_delta_count", 0)),
        1,
    )
    flip_rate = (
        float(diagnostics.get("steer_sign_flip_count", 0))
        / float(delta_count)
    )
    lag_n = max(
        int(diagnostics.get("camera_lag_count", 0)),
        1,
    )

    print(
        "  REWARD/SPEED | mean_reward={:+.5f} | target={:.3f} | "
        "cmd={:.3f} | measured={:.3f}".format(
            float(diagnostics["final_reward_sum"]) / n,
            float(target_mean),
            float(diagnostics.get("speed_cmd_sum", 0.0)) / n,
            float(diagnostics.get("speed_sum", 0.0)) / n,
        )
    )

    preview_n = max(
        int(
            diagnostics.get(
                "preview_lookahead_distance_count",
                0,
            )
        ),
        1,
    )

    print(
        "  ROAD | progress={:+.3f}m | lat={:.4f}/{:.4f}m mean/max | "
        "headingNow={:.2f}/{:.2f}deg | headingPreview={:.2f}/{:.2f}deg | "
        "lookahead={:.3f}m | previewValid={:.1f}%".format(
            float(diagnostics.get("progress_sum", 0.0)),
            float(diagnostics.get("abs_lat_sum", 0.0)) / n,
            float(diagnostics.get("abs_lat_max", 0.0)),
            np.degrees(float(diagnostics.get("abs_heading_sum", 0.0)) / n),
            np.degrees(float(diagnostics.get("abs_heading_max", 0.0))),
            np.degrees(
                float(
                    diagnostics.get(
                        "abs_preview_heading_sum",
                        0.0,
                    )
                )
                / n
            ),
            np.degrees(
                float(
                    diagnostics.get(
                        "abs_preview_heading_max",
                        0.0,
                    )
                )
            ),
            float(
                diagnostics.get(
                    "preview_lookahead_distance_sum",
                    0.0,
                )
            )
            / float(preview_n),
            100.0
            * float(
                diagnostics.get(
                    "preview_lookahead_distance_count",
                    0,
                )
            )
            / max(
                float(
                    diagnostics.get(
                        "samples",
                        0,
                    )
                ),
                1.0,
            ),
        )
    )

    print(
        "  STEER STABILITY | |steer|={:.3f} | |dSteer|={:.4f} mean max={:.4f} | "
        "signFlip={:.2f}% | >.50={:.1f}% >.75={:.1f}%".format(
            float(diagnostics.get("abs_steer_sum", 0.0)) / n,
            float(diagnostics.get("steer_delta_abs_sum", 0.0)) / float(delta_count),
            float(diagnostics.get("steer_delta_abs_max", 0.0)),
            100.0 * flip_rate,
            100.0 * float(diagnostics.get("steer_gt_050", 0)) / n,
            100.0 * float(diagnostics.get("steer_gt_075", 0)) / n,
        )
    )

    raw_delta_count = max(
        int(
            diagnostics.get(
                "raw_steer_delta_count",
                0,
            )
        ),
        1,
    )
    raw_flip_rate = (
        float(
            diagnostics.get(
                "raw_steer_sign_flip_count",
                0,
            )
        )
        / float(raw_delta_count)
    )
    limiter_hit_rate = (
        float(
            diagnostics.get(
                "steer_rate_limiter_hit_count",
                0,
            )
        )
        / n
    )

    print(
        "  STEER RAW | |raw|={:.3f} | |dRaw|={:.4f} mean max={:.4f} | "
        "rawFlip={:.2f}% | limiterHit={:.1f}%".format(
            float(
                diagnostics.get(
                    "abs_raw_steer_sum",
                    0.0,
                )
            )
            / n,
            float(
                diagnostics.get(
                    "raw_steer_delta_abs_sum",
                    0.0,
                )
            )
            / float(raw_delta_count),
            float(
                diagnostics.get(
                    "raw_steer_delta_abs_max",
                    0.0,
                )
            ),
            100.0 * raw_flip_rate,
            100.0 * limiter_hit_rate,
        )
    )

    teacher_strong_n = max(
        int(
            diagnostics.get(
                "teacher_strong_count",
                0,
            )
        ),
        1,
    )
    print(
        "  ORACLE TEACH | |target|={:.3f} | err={:.3f} | factor={:.3f} | "
        "strong(|target|>=.15) n={} err={:.3f}".format(
            float(
                diagnostics.get(
                    "teacher_abs_target_sum",
                    0.0,
                )
            )
            / n,
            float(
                diagnostics.get(
                    "teacher_abs_error_sum",
                    0.0,
                )
            )
            / n,
            float(
                diagnostics.get(
                    "teacher_factor_sum",
                    0.0,
                )
            )
            / n,
            int(
                diagnostics.get(
                    "teacher_strong_count",
                    0,
                )
            ),
            float(
                diagnostics.get(
                    "teacher_strong_abs_error_sum",
                    0.0,
                )
            )
            / float(
                teacher_strong_n
            ),
        )
    )

    print(
        "  CAMERA | lag_mean={:.2f}tick max={}tick".format(
            float(diagnostics.get("camera_lag_sum", 0.0)) / float(lag_n),
            int(diagnostics.get("camera_lag_max", 0)),
        )
    )

    mean_delta = (
        float(diagnostics.get("steer_delta_abs_sum", 0.0))
        / float(delta_count)
    )
    if flip_rate > 0.05 or mean_delta > 0.055:
        print(
            "  STABILITY WARNING | steering is still oscillatory; "
            "do NOT widen DR yet."
        )


def write_tensorboard_diagnostics(
    writer,
    diagnostics,
    global_step,
):
    if writer is None:
        return

    total_samples = max(
        int(
            diagnostics[
                "samples"
            ]
        ),
        1,
    )

    writer.add_scalar(
        "reward/final_mean",
        float(
            diagnostics[
                "final_reward_sum"
            ]
        )
        / total_samples,
        global_step,
    )

    for state_name in STATE_NAMES:
        values = diagnostics[
            "state"
        ][state_name]

        count = int(
            values["count"]
        )

        writer.add_scalar(
            "state/{}/occupancy".format(
                state_name
            ),
            float(count)
            / float(
                total_samples
            ),
            global_step,
        )

        if count > 0:
            writer.add_scalar(
                "state/{}/target_speed".format(
                    state_name
                ),
                _safe_mean(
                    values[
                        "target_sum"
                    ],
                    count,
                ),
                global_step,
            )

            writer.add_scalar(
                "state/{}/cmd_speed".format(
                    state_name
                ),
                _safe_mean(
                    values[
                        "cmd_sum"
                    ],
                    count,
                ),
                global_step,
            )

            writer.add_scalar(
                "state/{}/actual_speed".format(
                    state_name
                ),
                _safe_mean(
                    values[
                        "speed_sum"
                    ],
                    count,
                ),
                global_step,
            )

            writer.add_scalar(
                "state/{}/cmd_abs_error".format(
                    state_name
                ),
                _safe_mean(
                    values[
                        "abs_cmd_error_sum"
                    ],
                    count,
                ),
                global_step,
            )

            writer.add_scalar(
                "state/{}/speed_abs_error".format(
                    state_name
                ),
                _safe_mean(
                    values[
                        "abs_speed_error_sum"
                    ],
                    count,
                ),
                global_step,
            )

    for name, value in (
        diagnostics[
            "reward_term_sums"
        ].items()
    ):
        writer.add_scalar(
            "reward_term/{}".format(
                name
            ),
            float(value)
            / float(
                total_samples
            ),
            global_step,
        )

    writer.add_scalar(
        "recovery/trigger_events",
        float(
            diagnostics[
                "recovery_trigger_events"
            ]
        ),
        global_step,
    )

    writer.add_scalar(
        "recovery/success_events",
        float(
            diagnostics[
                "recovery_success_events"
            ]
        ),
        global_step,
    )


    n = float(total_samples)
    lag_count = max(int(diagnostics.get("camera_lag_count", 0)), 1)

    writer.add_scalar("behavior/update_progress_m", float(diagnostics.get("progress_sum", 0.0)), global_step)
    writer.add_scalar("behavior/update_avg_abs_lateral_m", float(diagnostics.get("abs_lat_sum", 0.0)) / n, global_step)
    writer.add_scalar("behavior/update_max_abs_lateral_m", float(diagnostics.get("abs_lat_max", 0.0)), global_step)
    writer.add_scalar("behavior/update_avg_abs_heading_deg", np.degrees(float(diagnostics.get("abs_heading_sum", 0.0)) / n), global_step)
    writer.add_scalar("behavior/update_max_abs_heading_deg", np.degrees(float(diagnostics.get("abs_heading_max", 0.0))), global_step)
    writer.add_scalar("control/update_avg_speed_mps", float(diagnostics.get("speed_sum", 0.0)) / n, global_step)
    writer.add_scalar("control/update_avg_speed_cmd_mps", float(diagnostics.get("speed_cmd_sum", 0.0)) / n, global_step)
    writer.add_scalar("control/update_avg_abs_steer", float(diagnostics.get("abs_steer_sum", 0.0)) / n, global_step)
    writer.add_scalar("control/steer_gt_050_rate", float(diagnostics.get("steer_gt_050", 0)) / n, global_step)
    writer.add_scalar("control/steer_gt_075_rate", float(diagnostics.get("steer_gt_075", 0)) / n, global_step)
    writer.add_scalar("curriculum/allowed_fraction", float(diagnostics.get("curriculum_allowed_samples", 0)) / n, global_step)
    writer.add_scalar("curriculum/injected_events", float(diagnostics.get("disturbance_trigger_events", 0)), global_step)
    writer.add_scalar("curriculum/injected_success_events", float(diagnostics.get("disturbance_success_events", 0)), global_step)
    writer.add_scalar("perf/camera_lag_mean_ticks", float(diagnostics.get("camera_lag_sum", 0.0)) / float(lag_count), global_step)
    writer.add_scalar("perf/camera_lag_max_ticks", float(diagnostics.get("camera_lag_max", 0)), global_step)


def stable_phase_config(global_step, args):
    """
    Controlled Sim2Real schedule for ONE map.

    Key idea: do NOT expose PPO to all perturbations at once.
    At most `dr_family_budget` families are active in a randomized episode.
    """
    step = int(global_step)

    if step < int(args.phase_clean_until):
        return {
            "name": "CLEAN_BASE",
            "prob": 0.0,
            "dynamics": False,
            "vision": False,
            "sensor": False,
        }

    if step < int(args.phase_vision_until):
        return {
            "name": "VISION_MILD",
            "prob": float(args.dr_prob_vision),
            "dynamics": False,
            "vision": bool(args.vision_dr),
            "sensor": False,
        }

    if step < int(args.phase_dynamics_until):
        return {
            "name": "VISION_OR_DYNAMICS",
            "prob": float(args.dr_prob_dynamics),
            "dynamics": bool(args.dynamics_dr),
            "vision": bool(args.vision_dr),
            "sensor": False,
        }

    if step < int(args.phase_controlled_until):
        return {
            "name": "CONTROLLED_ONE_FAMILY",
            "prob": float(args.dr_prob_controlled),
            "dynamics": bool(args.dynamics_dr),
            "vision": bool(args.vision_dr),
            "sensor": bool(args.sensor_dr),
        }

    return {
        "name": "NOMINAL_POLISH",
        "prob": float(args.dr_prob_polish),
        "dynamics": bool(args.dynamics_dr),
        "vision": bool(args.vision_dr),
        "sensor": bool(args.sensor_dr),
    }


def build_worker_configs(args):
    sources = [
        (
            int(args.port0),
            int(args.seed0),
        ),
        (
            int(args.port1),
            int(args.seed1),
        ),
    ]

    configs = []

    for worker_id in range(
        int(args.workers)
    ):
        port, seed = sources[
            worker_id
        ]

        configs.append(
            {
                "host": args.host,
                "port": int(
                    port
                ),
                "seed": int(
                    seed
                ),
                "carla_timeout": float(
                    args.carla_timeout
                ),
                "expected_map": (
                    args.expected_map
                ),
                "safe_spawns": [
                    int(x) for x in args.safe_spawns
                ],
                "desired_speed": float(
                    args.desired_speed
                ),
                "steer_rate_limit": float(args.steer_rate_limit),
                "speed_rate_limit": float(args.speed_rate_limit),
                "steer_deadband": float(args.steer_deadband),
                "dr_family_budget": int(args.dr_family_budget),
                "max_episode_seconds": float(
                    args.max_episode_seconds
                ),

                "encoder_device": (
                    args.encoder_device
                ),
                "worker_ppo_device": (
                    args.worker_ppo_device
                ),
                "worker_torch_threads": int(
                    args.worker_torch_threads
                ),
                "action_std_init": float(
                    args.action_std_init
                ),

                "dynamics_dr": bool(
                    args.dynamics_dr
                ),
                "vision_dr": bool(
                    args.vision_dr
                ),
                "sensor_dr": bool(
                    args.sensor_dr
                ),
                "recovery_scenarios": bool(
                    args.recovery_scenarios
                ),

                "disturbance_prob": float(
                    args.disturbance_prob
                ),
                "disturbance_duration_min_s": float(
                    args.disturbance_duration_min_s
                ),
                "disturbance_duration_max_s": float(
                    args.disturbance_duration_max_s
                ),
                "disturbance_steer_min": float(
                    args.disturbance_steer_min
                ),
                "disturbance_steer_max": float(
                    args.disturbance_steer_max
                ),
                "recovery_min_episode_step": int(
                    args.recovery_min_episode_step
                ),
                "recovery_stable_ticks": int(
                    args.recovery_stable_ticks
                ),
                "recovery_stable_lateral_m": float(
                    args.recovery_stable_lateral_m
                ),
                "recovery_stable_heading_deg": float(
                    args.recovery_stable_heading_deg
                ),
                "recovery_stable_speed_mps": float(
                    args.recovery_stable_speed_mps
                ),
                "recovery_cooldown_ticks": int(
                    args.recovery_cooldown_ticks
                ),
                "recovery_max_events": int(
                    args.recovery_max_events
                ),
            }
        )

    return configs


def verify_ready_workers(
    ready_messages,
):
    if not ready_messages:
        raise RuntimeError(
            "No worker is ready."
        )

    base_profile = (
        ready_messages[0][
            "reward_profile"
        ]
    )

    verify_reward_profile(
        base_profile
    )

    for message in (
        ready_messages[1:]
    ):
        verify_reward_profile(
            message[
                "reward_profile"
            ]
        )

        # STABLE035 reward_profile contains only the original-style
        # speed thresholds: min / target / max.  Recovery-state labels
        # are diagnostics only and are intentionally NOT reward-profile
        # keys.
        for profile_key in ("min", "target", "max"):
            if abs(
                float(
                    message[
                        "reward_profile"
                    ][profile_key]
                )
                - float(
                    base_profile[
                        profile_key
                    ]
                )
            ) > 1e-9:
                raise RuntimeError(
                    "Workers have different reward profiles."
                )


def print_startup(
    args,
    current_step,
    target_step,
    loaded,
    ready_messages,
    current_action_std,
):
    print(
        "=" * 112
    )
    print(
        "PPO V5 SINGLE-MAP STABLE035 | NEW RUN"
        if not args.resume
        else
        "PPO V5 SINGLE-MAP STABLE035 | RESUME"
    )
    print(
        "=" * 112
    )

    print(
        "model namespace    :",
        MODEL_NAME,
    )
    print(
        "checkpoint version : PPO_V5_CLEAN"
    )
    print(
        "trainer version    :",
        TRAINER_VERSION,
    )
    print(
        "loaded checkpoint  :",
        (
            loaded
            if loaded is not None
            else "NONE - RANDOM INIT"
        ),
    )
    print(
        "global step        :",
        current_step,
    )
    print(
        "target step        :",
        (
            "SMOKE ONLY"
            if args.smoke_test
            else target_step
        ),
    )
    if args.smoke_test:
        effective_probe_std = (
            float(args.probe_action_std)
            if float(args.probe_action_std) > 0.0
            else float(current_action_std)
        )
        print(
            "probe action std  :",
            "{:.6f}".format(
                effective_probe_std
            ),
            (
                "(near-deterministic)"
                if float(args.probe_action_std) > 0.0
                else "(loaded stochastic std)"
            ),
        )
    if (
        bool(args.resume)
        and not bool(args.smoke_test)
        and float(args.resume_action_std) > 0.0
    ):
        print(
            "resume train std  :",
            "{:.4f}".format(
                float(current_action_std)
            ),
        )
    print(
        "workers            :",
        args.workers,
    )
    print(
        "rollout total      :",
        (
            int(
                args.smoke_steps_per_worker
            )
            * int(
                args.workers
            )
            if args.smoke_test
            else args.rollout_total
        ),
    )
    profile = ready_messages[0]["reward_profile"]

    print(
        "speed profile      : MIN={:.3f} | TARGET={:.3f} | HARD MAX={:.3f} m/s".format(
            profile["min"],
            profile["target"],
            profile["max"],
        )
    )
    print(
        "steer stability    : rate_limit={:.2f}/s | deadband={:.3f} | speed_slew={:.2f}m/s^2".format(
            float(args.steer_rate_limit),
            float(args.steer_deadband),
            float(args.speed_rate_limit),
        )
    )

    print(
        "DR                 : dynamics={} | vision={} | sensor={}".format(
            bool(
                args.dynamics_dr
            ),
            bool(
                args.vision_dr
            ),
            bool(
                args.sensor_dr
            ),
        )
    )

    print(
        "ONE map spawns     : {} (balanced; both workers)".format(
            list(args.safe_spawns)
        )
    )
    print(
        "spawn perturbation : HARD OFF | offset=0 | heading=0"
    )
    print(
        "recovery injection : HARD OFF"
    )
    print(
        "controlled DR      : clean<{} | vision<{} | vis/dyn<{} | one-family<{} | polish after".format(
            int(args.phase_clean_until),
            int(args.phase_vision_until),
            int(args.phase_dynamics_until),
            int(args.phase_controlled_until),
        )
    )
    print(
        "DR family budget   : {} family/episode max".format(
            int(args.dr_family_budget)
        )
    )

    print(
        "policy sharing     : ONE shared PPO snapshot per rollout"
    )
    print(
        "return boundaries  : per-worker truncation + V(next_obs)"
    )

    for message in ready_messages:
        print(
            "worker {}           : port={} seed={} map={} Hz={:.1f} dr_seed={}".format(
                message[
                    "worker"
                ],
                message[
                    "port"
                ],
                message[
                    "seed"
                ],
                message[
                    "map"
                ],
                message[
                    "control_hz"
                ],
                message[
                    "dr_seed"
                ],
            )
        )

    print(
        "=" * 112
    )


def main():
    args = parse_args()
    validate_args(args)

    mp.freeze_support()

    torch.set_num_threads(
        int(
            args.learner_torch_threads
        )
    )

    seed_everything(
        int(SEED)
    )

    master = PPOAgent(
        town=MODEL_NAME,
        action_std_init=float(
            args.action_std_init
        ),
        device=args.learner_device,
    )

    (
        current_step,
        current_episode,
        loaded_checkpoint,
    ) = prepare_fresh_or_resume(
        agent=master,
        args=args,
    )

    master.memory.clear()

    if (
        bool(args.resume)
        and not bool(args.smoke_test)
        and float(args.resume_action_std) > 0.0
    ):
        loaded_std = float(
            master.action_std
        )
        master.set_action_std(
            float(args.resume_action_std)
        )
        print(
            "RESUME ACTION STD OVERRIDE | loaded={:.4f} -> train={:.4f}".format(
                loaded_std,
                float(master.action_std),
            )
        )

    target_step = int(
        args.total_timesteps
    )

    if (
        bool(args.resume)
        and current_step
        >= target_step
        and not args.smoke_test
    ):
        print(
            "Training already complete | step={}/{}".format(
                current_step,
                target_step,
            )
        )
        return

    worker_configs = (
        build_worker_configs(
            args
        )
    )

    ctx = mp.get_context(
        "spawn"
    )

    parent_conns = []
    processes = []

    writer = make_writer(
        args
    )

    run_start = (
        time.time()
    )

    global_step = int(
        current_step
    )
    episode = int(
        current_episode
    )

    failure_tracker = (
        FailureClusterTracker.load()
        if bool(args.resume)
        else FailureClusterTracker()
    )

    try:
        for worker_id in range(
            int(args.workers)
        ):
            (
                parent_conn,
                child_conn,
            ) = ctx.Pipe(
                duplex=True
            )

            process = ctx.Process(
                target=worker_main,
                args=(
                    worker_id,
                    child_conn,
                    worker_configs[
                        worker_id
                    ],
                ),
                name=(
                    "CARLA-V5-STABLE035-W{}".format(
                        worker_id
                    )
                ),
            )

            process.daemon = False
            process.start()

            child_conn.close()

            parent_conns.append(
                parent_conn
            )
            processes.append(
                process
            )

        ready = []

        for worker_id, conn in enumerate(
            parent_conns
        ):
            if not conn.poll(
                float(
                    args.worker_timeout
                )
            ):
                raise TimeoutError(
                    "Worker {} init timeout after {} s.".format(
                        worker_id,
                        args.worker_timeout,
                    )
                )

            message = conn.recv()

            if message.get(
                "type"
            ) == "error":
                raise RuntimeError(
                    "Worker {} init failed:\n{}".format(
                        worker_id,
                        message.get(
                            "traceback",
                            message.get(
                                "error"
                            ),
                        ),
                    )
                )

            if message.get(
                "type"
            ) != "ready":
                raise RuntimeError(
                    "Unexpected worker init message: {}".format(
                        message
                    )
                )

            ready.append(
                message
            )

        verify_ready_workers(
            ready
        )

        print_startup(
            args=args,
            current_step=current_step,
            target_step=target_step,
            loaded=loaded_checkpoint,
            ready_messages=ready,
            current_action_std=float(
                master.action_std
            ),
        )

        decay_bucket = int(
            global_step
            // int(
                args.action_std_decay_freq
            )
        )

        next_checkpoint_step = (
            (
                global_step
                // int(
                    args.checkpoint_every_steps
                )
            )
            + 1
        ) * int(
            args.checkpoint_every_steps
        )

        update_index = 0

        while (
            bool(args.smoke_test)
            or global_step
            < target_step
        ):
            update_index += 1

            if args.smoke_test:
                requested_total = (
                    int(
                        args.smoke_steps_per_worker
                    )
                    * int(
                        args.workers
                    )
                )
            else:
                requested_total = min(
                    int(
                        args.rollout_total
                    ),
                    int(
                        target_step
                        - global_step
                    ),
                )

            per_worker = split_steps(
                requested_total,
                int(
                    args.workers
                ),
            )

            policy_snapshot = (
                state_dict_to_numpy(
                    master.old_policy
                )
            )

            collect_start = (
                time.time()
            )

            active_workers = []
            phase_cfg = stable_phase_config(
                global_step,
                args,
            )

            for worker_id, steps in enumerate(
                per_worker
            ):
                if int(steps) <= 0:
                    continue

                parent_conns[
                    worker_id
                ].send(
                    {
                        "cmd": "collect",
                        "steps": int(
                            steps
                        ),
                        "policy_state": (
                            policy_snapshot
                        ),
                        "action_std": float(
                            args.probe_action_std
                            if (
                                bool(args.smoke_test)
                                and float(args.probe_action_std) > 0.0
                            )
                            else master.action_std
                        ),
                        "desired_speed": float(
                            args.desired_speed
                        ),
                        "phase_name": str(phase_cfg["name"]),
                        "dr_probability": float(phase_cfg["prob"]),
                        "dr_dynamics": bool(phase_cfg["dynamics"]),
                        "dr_vision": bool(phase_cfg["vision"]),
                        "dr_sensor": bool(phase_cfg["sensor"]),
                        "dr_family_budget": int(args.dr_family_budget),
                    }
                )

                active_workers.append(
                    worker_id
                )

            messages = []

            for worker_id in (
                active_workers
            ):
                conn = parent_conns[
                    worker_id
                ]

                if not conn.poll(
                    float(
                        args.worker_timeout
                    )
                ):
                    raise TimeoutError(
                        "Worker {} rollout timeout after {} s.".format(
                            worker_id,
                            args.worker_timeout,
                        )
                    )

                message = conn.recv()

                if message.get(
                    "type"
                ) == "error":
                    raise RuntimeError(
                        "Worker {} failed:\n{}".format(
                            worker_id,
                            message.get(
                                "traceback",
                                message.get(
                                    "error"
                                ),
                            ),
                        )
                    )

                if message.get(
                    "type"
                ) != "rollout":
                    raise RuntimeError(
                        "Unexpected worker message: {}".format(
                            message
                        )
                    )

                messages.append(
                    message
                )

            collect_wall = (
                time.time()
                - collect_start
            )

            total_samples = sum(
                int(
                    message[
                        "samples"
                    ]
                )
                for message in messages
            )

            if total_samples != int(
                requested_total
            ):
                raise RuntimeError(
                    "Expected {} samples, got {}.".format(
                        requested_total,
                        total_samples,
                    )
                )

            if not args.smoke_test:
                event_step = int(global_step) + int(requested_total)
                for message in messages:
                    for event in message.get("failure_events", []):
                        failure_tracker.add(
                            event,
                            global_step=event_step,
                        )

            diagnostics = (
                aggregate_diagnostics(
                    messages
                )
            )

            if args.smoke_test:
                print(
                    "\nSMOKE STABLE035 PASS | rollout collection only"
                )

                for message in messages:
                    print(
                        "  W{} | samples={} | wall={:.2f}s | SPS={:.2f} | episodes={}".format(
                            message[
                                "worker"
                            ],
                            message[
                                "samples"
                            ],
                            message[
                                "wall_s"
                            ],
                            float(
                                message[
                                    "samples"
                                ]
                            )
                            / max(
                                float(
                                    message[
                                        "wall_s"
                                    ]
                                ),
                                1e-9,
                            ),
                            len(
                                message[
                                    "episodes"
                                ]
                            ),
                        )
                    )

                print_state_diagnostics(
                    diagnostics
                )

                print(
                    "NO PPO UPDATE | NO CHECKPOINT | NO TRAINING STATE CHANGE"
                )
                break

            # ------------------------------------------------------
            # Merge on-policy worker rollouts into master.
            # ------------------------------------------------------
            master.memory.clear()
            merged = 0

            for message in sorted(
                messages,
                key=lambda item: int(
                    item["worker"]
                ),
            ):
                merged += (
                    append_rollout_to_master(
                        master,
                        message[
                            "rollout"
                        ],
                    )
                )

            if merged != int(
                requested_total
            ):
                raise RuntimeError(
                    "Merged {} != requested {}.".format(
                        merged,
                        requested_total,
                    )
                )

            # Every worker ends at an independent return boundary.
            ppo_metrics = master.learn(
                last_obs=None
            )

            global_step += int(
                requested_total
            )

            episode_metrics = (
                aggregate_episode_metrics(
                    messages
                )
            )

            episode += int(
                episode_metrics[
                    "episodes"
                ]
            )

            decay_bucket = (
                maybe_decay_action_std(
                    agent=master,
                    global_step=global_step,
                    previous_bucket=(
                        decay_bucket
                    ),
                    args=args,
                )
            )

            combined_sps = (
                float(
                    requested_total
                )
                / max(
                    collect_wall,
                    1e-9,
                )
            )

            elapsed_min = (
                time.time()
                - run_start
            ) / 60.0

            print(
                "\nUPDATE {:05d} | step={}/{} | samples={} | "
                "collect={:.1f}s | SPS={:.2f} | action_std={:.4f} | "
                "elapsed={:.1f}m".format(
                    update_index,
                    global_step,
                    target_step,
                    requested_total,
                    collect_wall,
                    combined_sps,
                    master.action_std,
                    elapsed_min,
                )
            )

            print(
                "  PHASE | {} | DR episode={:.0f}% | dynamics={} vision={} sensor={} | budget={}".format(
                    str(phase_cfg["name"]),
                    100.0 * float(phase_cfg["prob"]),
                    bool(phase_cfg["dynamics"]),
                    bool(phase_cfg["vision"]),
                    bool(phase_cfg["sensor"]),
                    int(args.dr_family_budget),
                )
            )

            print(
                "  PPO | loss={:+.5f} | policy={:+.5f} | value={:.5f} | "
                "entropy={:.5f} | mean_return={:+.5f}".format(
                    float(
                        ppo_metrics.get(
                            "loss",
                            0.0,
                        )
                    ),
                    float(
                        ppo_metrics.get(
                            "policy_loss",
                            0.0,
                        )
                    ),
                    float(
                        ppo_metrics.get(
                            "value_loss",
                            0.0,
                        )
                    ),
                    float(
                        ppo_metrics.get(
                            "entropy",
                            0.0,
                        )
                    ),
                    float(
                        ppo_metrics.get(
                            "mean_return",
                            0.0,
                        )
                    ),
                )
            )

            print_state_diagnostics(
                diagnostics
            )

            if int(episode_metrics.get("episodes", 0)) > 0:
                print(
                    "  EPISODE HEALTH | n={} reward={:+.2f} steps={:.1f} progress={:.2f}m | "
                    "lat_mean={:.4f}m lat_max={:.4f}m | head_mean={:.2f}deg | "
                    "offroad={:.1f}% time_limit={:.1f}%".format(
                        int(episode_metrics["episodes"]),
                        float(episode_metrics.get("avg_reward", 0.0)),
                        float(episode_metrics.get("avg_steps", 0.0)),
                        float(episode_metrics.get("avg_progress_m", 0.0)),
                        float(episode_metrics.get("avg_abs_lateral_m", 0.0)),
                        float(episode_metrics.get("max_abs_lateral_m", 0.0)),
                        float(episode_metrics.get("avg_abs_heading_deg", 0.0)),
                        100.0 * float(episode_metrics.get("offroad_rate", 0.0)),
                        100.0 * float(episode_metrics.get("time_limit_rate", 0.0)),
                    )
                )

            print(
                "  COMPLETED EPISODES | count={}".format(
                    episode_metrics["episodes"]
                )
            )

            if writer is not None:
                for key, value in (
                    ppo_metrics.items()
                ):
                    writer.add_scalar(
                        "ppo/{}".format(
                            key
                        ),
                        float(
                            value
                        ),
                        global_step,
                    )

                writer.add_scalar(
                    "ppo/action_std",
                    float(
                        master.action_std
                    ),
                    global_step,
                )

                writer.add_scalar(
                    "perf/combined_sps",
                    float(
                        combined_sps
                    ),
                    global_step,
                )

                writer.add_scalar(
                    "perf/collection_wall_s",
                    float(
                        collect_wall
                    ),
                    global_step,
                )

                writer.add_scalar(
                    "episode/completed",
                    float(
                        episode_metrics[
                            "episodes"
                        ]
                    ),
                    global_step,
                )

                writer.add_scalar(
                    "episode/recovery_rate",
                    float(
                        episode_metrics[
                            "recovery_rate"
                        ]
                    ),
                    global_step,
                )

                if int(episode_metrics.get("episodes", 0)) > 0:
                    writer.add_scalar("behavior/episode_reward_mean", float(episode_metrics.get("avg_reward", 0.0)), global_step)
                    writer.add_scalar("behavior/episode_steps_mean", float(episode_metrics.get("avg_steps", 0.0)), global_step)
                    writer.add_scalar("behavior/progress_m_mean", float(episode_metrics.get("avg_progress_m", 0.0)), global_step)
                    writer.add_scalar("behavior/avg_abs_lateral_m", float(episode_metrics.get("avg_abs_lateral_m", 0.0)), global_step)
                    writer.add_scalar("behavior/max_abs_lateral_m", float(episode_metrics.get("max_abs_lateral_m", 0.0)), global_step)
                    writer.add_scalar("behavior/avg_abs_heading_deg", float(episode_metrics.get("avg_abs_heading_deg", 0.0)), global_step)
                    writer.add_scalar("behavior/max_abs_heading_deg", float(episode_metrics.get("max_abs_heading_deg", 0.0)), global_step)
                    writer.add_scalar("behavior/offroad_rate", float(episode_metrics.get("offroad_rate", 0.0)), global_step)
                    writer.add_scalar("behavior/collision_rate", float(episode_metrics.get("collision_rate", 0.0)), global_step)
                    writer.add_scalar("behavior/stuck_rate", float(episode_metrics.get("stuck_rate", 0.0)), global_step)
                    writer.add_scalar("behavior/time_limit_rate", float(episode_metrics.get("time_limit_rate", 0.0)), global_step)
                    writer.add_scalar("control/avg_speed_mps", float(episode_metrics.get("avg_speed_mps", 0.0)), global_step)
                    writer.add_scalar("control/avg_speed_cmd_mps", float(episode_metrics.get("avg_speed_cmd_mps", 0.0)), global_step)
                    writer.add_scalar("control/avg_abs_steer", float(episode_metrics.get("avg_abs_steer", 0.0)), global_step)
                    writer.add_scalar("curriculum/injected_success_rate", float(episode_metrics.get("disturbance_success_rate", 0.0)), global_step)

                write_tensorboard_diagnostics(
                    writer=writer,
                    diagnostics=(
                        diagnostics
                    ),
                    global_step=(
                        global_step
                    ),
                )

            if (
                global_step
                >= next_checkpoint_step
                or global_step
                >= target_step
            ):
                save_checkpoint(
                    agent=master,
                    global_step=(
                        global_step
                    ),
                    episode=episode,
                    args=args,
                )
                failure_tracker.save()
                failure_tracker.print_report(limit=8)

                while (
                    next_checkpoint_step
                    <= global_step
                ):
                    next_checkpoint_step += int(
                        args.checkpoint_every_steps
                    )

            if global_step >= target_step:
                print(
                    "\nV5 SINGLE-MAP STABLE035 COMPLETE | step={} | episodes={} | elapsed={:.1f} min".format(
                        global_step,
                        episode,
                        elapsed_min,
                    )
                )
                break

    except KeyboardInterrupt:
        print(
            "\nStopped by user."
        )

        if not args.smoke_test:
            try:
                path = save_checkpoint(
                    agent=master,
                    global_step=(
                        global_step
                    ),
                    episode=episode,
                    args=args,
                )

                failure_tracker.save()
                failure_tracker.print_report(limit=8)

                print(
                    "STOP CHECKPOINT:",
                    path,
                )

            except Exception:
                logging.exception(
                    "Could not save stop checkpoint."
                )

    except Exception:
        logging.exception(
            "PPO V5 SINGLE-MAP STABLE035 trainer failed."
        )
        raise

    finally:
        for conn in parent_conns:
            try:
                conn.send(
                    {
                        "cmd": "close"
                    }
                )
            except Exception:
                pass

        for process in processes:
            try:
                process.join(
                    timeout=10.0
                )
            except Exception:
                pass

            if process.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass

        for conn in parent_conns:
            try:
                conn.close()
            except Exception:
                pass

        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass

        print(
            "V5 CLEAN cleanup complete."
        )


if __name__ == "__main__":
    main()


