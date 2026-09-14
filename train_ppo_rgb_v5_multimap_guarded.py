# -*- coding: utf-8 -*-
"""
PPO V5 MULTI-OVAL GUARDED TRAINER
=================================

Design goals
------------
1) Train PPO on multiple OVAL maps instead of one fixed map.
2) Keep spawn reset CLEAN: original CARLA spawn position + original rotation.
3) Use 4 balanced spawns per map. No lateral/heading perturbation at reset.
4) Recovery is DELAYED and only injected after a stable nominal window.
5) Recovery episode selection defaults to 30%, therefore 70% pure nominal.
6) Disturbance sign is balanced by the environment (+/- in shuffled pairs).
7) Two anchor maps are always the reference task:
       /Game/maptotrai/mapto_trai
       /Game/maptophai/mapto_phai
8) Auxiliary oval maps improve visual/generalization diversity.
9) /Game/maptrangcorao/maptuong is left-turn-only, so it receives low weight.
10) A deterministic CANARY evaluator checks BOTH anchor maps x 4 spawns.
11) A failing candidate is rejected and PPO is rolled back to the last CHAMPION.
12) Curriculum advances only after repeated canary PASS results.
13) MAIN numbered checkpoints are created ONLY after CANARY PASS.
14) CANARY FAIL snapshots live only under rejected/.
15) Ctrl+C saves only an emergency/ snapshot; it never promotes a champion.

Recommended fresh run
---------------------
PowerShell:

python .\train_ppo_rgb_v5_multimap_guarded.py `
  --workers 2 `
  --total-timesteps 2000000 `
  --rollout-total 1024 `
  --map-block-steps 10000 `
  --canary-every-steps 20000 `
  --canary-max-steps 750 `
  --learner-device cpu `
  --worker-ppo-device cpu `
  --encoder-device cpu

Optional fine-tune from an existing PPO_V5_CLEAN checkpoint:

python .\train_ppo_rgb_v5_multimap_guarded.py `
  --workers 2 `
  --init-checkpoint ".\\preTrained_models\\ppo\\...\\ppo_policy_21_.pth" `
  --total-timesteps 1000000 `
  --rollout-total 1024 `
  --learner-device cpu `
  --worker-ppo-device cpu `
  --encoder-device cpu

Python 3.7+.
"""

from __future__ import print_function

import argparse
import copy
import json
import logging
import math
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
    SEED,
    TOTAL_TIMESTEPS,
)
from simulation.carla_connection_v5 import carla
from simulation.carla_environment_rgb_v5_multimap import (
    CarlaEnvironmentRGBV5MultiMap,
    configure_recovery_scenarios_v5,
)
from vehicle_specs_v5 import CARLA_GEOMETRY_SCALE

# Reuse the already-tested rollout/diagnostic helpers from the current V5 trainer.
# No training logic is delegated to that trainer; only pure helper classes/functions.
from train_ppo_rgb_v5 import (
    EpisodeAccumulator,
    RolloutDiagnostics,
    aggregate_diagnostics,
    aggregate_episode_metrics,
    append_rollout_to_master,
    export_memory,
    load_numpy_state,
    print_state_diagnostics,
    seed_everything,
    split_steps,
    state_dict_to_numpy,
    write_tensorboard_diagnostics,
)


# ============================================================================
# Fixed task map set
# ============================================================================

ANCHOR_MAPS = (
    "/Game/maptotrai/mapto_trai",
    "/Game/maptophai/mapto_phai",
)

AUX_OVAL_MAPS = (
    "/Game/dancu_phai/dancu_phai",
    "/Game/dancu_trai/dancu_trai",
    "/Game/maplonxon/maplonxon",
    "/Game/maplonxon1/maplonxon1",
    "/Game/rungphai/rung_phai",
    "/Game/rungtrai/rung_trai",
)

LEFT_ONLY_STRESS_MAP = "/Game/maptrangcorao/maptuong"

# Exact weighted cycle for Stage 2/3:
#   anchor 10/17 = 58.8%
#   six auxiliary 6/17 = 35.3%
#   left-only stress 1/17 = 5.9%
MULTIMAP_WEIGHTED_SLOTS = (
    [ANCHOR_MAPS[0]] * 5
    + [ANCHOR_MAPS[1]] * 5
    + list(AUX_OVAL_MAPS)
    + [LEFT_ONLY_STRESS_MAP]
)

MODEL_NAME = "automav5_rgb_multimap_guarded"
STATE_NAME = "training_state_multimap_guarded_v5.json"
CHAMPION_NAME = "champion_multimap_guarded_v5.pth"
TRAINER_VERSION = "TRAINER_PPO_V5_MULTIMAP_GUARDED_2"

CONTROL_HZ = 50.0

STAGE_NAMES = {
    0: "ANCHOR_NOMINAL",
    1: "ANCHOR_DELAYED_RECOVERY",
    2: "MULTIMAP_DELAYED_RECOVERY",
    3: "MULTIMAP_FULL_SIM2REAL",
}


# ============================================================================
# CLI / validation
# ============================================================================

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
            raise argparse.ArgumentTypeError("Spawn numbers must be 1-based and > 0.")
        values.append(number)
    values = list(dict.fromkeys(values))
    if not values:
        raise argparse.ArgumentTypeError("At least one spawn is required.")
    return values


def parse_args():
    p = argparse.ArgumentParser(
        description="PPO V5 multi-oval trainer with balanced clean spawns and guardrail rollback."
    )

    # CARLA workers.
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port0", type=int, default=2000)
    p.add_argument("--port1", type=int, default=2010)
    p.add_argument("--carla-timeout", type=float, default=180.0)
    p.add_argument("--worker-timeout", type=float, default=1200.0)

    # Fresh / resume.
    p.add_argument("--resume", type=boolean_string, default=False)
    p.add_argument("--init-checkpoint", type=str, default="")
    p.add_argument("--total-timesteps", type=int, default=int(TOTAL_TIMESTEPS))
    p.add_argument("--rollout-total", type=int, default=1024)
    p.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=int(PPO_CHECKPOINT_EVERY_STEPS),
        help=(
            "DEPRECATED/IGNORED in GUARDED V2. Main numbered checkpoints are "
            "saved only after CANARY PASS."
        ),
    )
    p.add_argument(
        "--keep-rejected",
        type=int,
        default=5,
        help="Keep only the newest N rejected diagnostic snapshots.",
    )
    p.add_argument(
        "--keep-emergency",
        type=int,
        default=3,
        help="Keep only the newest N Ctrl+C/final-unverified emergency snapshots.",
    )

    # Clean spawn task.
    p.add_argument(
        "--safe-spawns",
        type=parse_spawn_numbers,
        default=parse_spawn_numbers("1,2,3,4"),
    )
    p.add_argument("--desired-speed", type=float, default=1.0)
    p.add_argument("--max-episode-seconds", type=float, default=60.0)

    # Map scheduling. Block switching avoids reloading CARLA every rollout.
    p.add_argument(
        "--map-block-steps",
        type=int,
        default=10000,
        help="Total transitions across workers before the scheduler may change maps.",
    )

    # Delayed recovery. Pure nominal fraction is 1 - disturbance_prob.
    p.add_argument("--recovery-episode-prob", type=float, default=0.30)
    p.add_argument("--disturbance-duration-min-s", type=float, default=0.10)
    p.add_argument("--disturbance-duration-max-s", type=float, default=0.16)
    p.add_argument("--disturbance-steer-min", type=float, default=0.10)
    p.add_argument("--disturbance-steer-max", type=float, default=0.18)
    p.add_argument("--recovery-min-episode-step", type=int, default=700)
    p.add_argument("--recovery-stable-ticks", type=int, default=100)
    p.add_argument("--recovery-stable-lateral-m", type=float, default=0.040)
    p.add_argument("--recovery-stable-heading-deg", type=float, default=5.0)
    p.add_argument("--recovery-stable-speed-mps", type=float, default=0.40)
    p.add_argument("--recovery-cooldown-ticks", type=int, default=400)
    p.add_argument("--recovery-max-events", type=int, default=1)

    # Final-stage DR. Stage 0/1/2 keep these OFF automatically.
    p.add_argument("--dynamics-dr", type=boolean_string, default=True)
    p.add_argument("--vision-dr", type=boolean_string, default=True)
    p.add_argument("--sensor-dr", type=boolean_string, default=True)

    # Automatic stage gates.
    p.add_argument("--stage0-min-steps", type=int, default=100000)
    p.add_argument("--stage1-min-steps", type=int, default=150000)
    p.add_argument("--stage2-min-steps", type=int, default=200000)
    p.add_argument(
        "--stage-pass-count",
        type=int,
        default=2,
        help="Consecutive canary passes required before advancing a stage.",
    )

    # Canary guardrail.
    p.add_argument("--canary-every-steps", type=int, default=20000)
    p.add_argument(
        "--canary-max-steps",
        type=int,
        default=750,
        help="Per spawn. 750 ticks = 15 s at 50 Hz.",
    )
    p.add_argument("--canary-avg-lateral-max-m", type=float, default=0.025)
    p.add_argument("--canary-max-lateral-max-m", type=float, default=0.060)
    p.add_argument("--canary-avg-heading-max-deg", type=float, default=6.0)
    p.add_argument("--canary-edge-ratio-threshold", type=float, default=0.70)
    p.add_argument("--canary-edge-rate-max", type=float, default=0.05)
    p.add_argument("--canary-steer075-rate-max", type=float, default=0.08)
    p.add_argument("--canary-min-progress-m", type=float, default=2.0)
    p.add_argument("--canary-center-start-ratio", type=float, default=0.15)
    p.add_argument("--canary-center-escape-ratio", type=float, default=0.55)
    p.add_argument("--canary-center-escape-window-ticks", type=int, default=100)
    p.add_argument(
        "--canary-regression-factor",
        type=float,
        default=2.0,
        help="Reject anchor avg-lateral regression versus champion beyond factor + slack.",
    )
    p.add_argument("--canary-regression-slack-m", type=float, default=0.003)

    # Exploration / devices.
    p.add_argument("--seed0", type=int, default=505)
    p.add_argument("--seed1", type=int, default=1505)
    p.add_argument("--learner-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--worker-ppo-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--worker-torch-threads", type=int, default=2)
    p.add_argument("--learner-torch-threads", type=int, default=2)
    p.add_argument("--action-std-init", type=float, default=float(ACTION_STD_INIT))
    p.add_argument("--action-std-min", type=float, default=float(PPO_ACTION_STD_MIN))
    p.add_argument("--action-std-decay", type=float, default=float(PPO_ACTION_STD_DECAY))
    p.add_argument(
        "--action-std-decay-freq",
        type=int,
        default=int(PPO_ACTION_STD_DECAY_FREQ),
    )

    p.add_argument("--tensorboard", type=boolean_string, default=True)
    p.add_argument("--smoke-test", type=boolean_string, default=False)
    p.add_argument("--smoke-steps-per-worker", type=int, default=64)

    return p.parse_args()


def validate_args(a):
    if int(a.workers) == 2 and int(a.port0) == int(a.port1):
        raise ValueError("port0 and port1 must differ.")
    if bool(a.resume) and str(a.init_checkpoint).strip():
        raise ValueError("Use either --resume true OR --init-checkpoint, not both.")
    if int(a.total_timesteps) <= 0:
        raise ValueError("--total-timesteps must be > 0.")
    if int(a.rollout_total) < int(a.workers):
        raise ValueError("--rollout-total must be >= worker count.")
    if int(a.map_block_steps) <= 0:
        raise ValueError("--map-block-steps must be > 0.")
    if abs(float(a.desired_speed) - 1.0) > 1e-9:
        raise ValueError("V5 speed profile is locked to --desired-speed 1.0.")
    if not (0.0 <= float(a.recovery_episode_prob) <= 1.0):
        raise ValueError("--recovery-episode-prob must be in [0,1].")
    if int(a.canary_every_steps) <= 0 or int(a.canary_max_steps) <= 0:
        raise ValueError("Canary intervals must be > 0.")
    if int(a.stage_pass_count) <= 0:
        raise ValueError("--stage-pass-count must be > 0.")
    if int(a.worker_torch_threads) <= 0 or int(a.learner_torch_threads) <= 0:
        raise ValueError("Torch thread counts must be > 0.")
    if int(a.keep_rejected) < 0:
        raise ValueError("--keep-rejected must be >= 0.")
    if int(a.keep_emergency) < 0:
        raise ValueError("--keep-emergency must be >= 0.")


# ============================================================================
# Curriculum / map scheduler
# ============================================================================

def stage_config(stage, args):
    stage = int(stage)
    if stage not in STAGE_NAMES:
        raise ValueError("Unknown curriculum stage {}".format(stage))

    if stage == 0:
        return {
            "name": STAGE_NAMES[stage],
            "multimap": False,
            "recovery": False,
            "recovery_prob": 0.0,
            "dynamics_dr": False,
            "vision_dr": False,
            "sensor_dr": False,
        }

    if stage == 1:
        return {
            "name": STAGE_NAMES[stage],
            "multimap": False,
            "recovery": True,
            "recovery_prob": float(args.recovery_episode_prob),
            "dynamics_dr": False,
            "vision_dr": False,
            "sensor_dr": False,
        }

    if stage == 2:
        return {
            "name": STAGE_NAMES[stage],
            "multimap": True,
            "recovery": True,
            "recovery_prob": float(args.recovery_episode_prob),
            "dynamics_dr": False,
            "vision_dr": False,
            "sensor_dr": False,
        }

    return {
        "name": STAGE_NAMES[stage],
        "multimap": True,
        "recovery": True,
        "recovery_prob": float(args.recovery_episode_prob),
        "dynamics_dr": bool(args.dynamics_dr),
        "vision_dr": bool(args.vision_dr),
        "sensor_dr": bool(args.sensor_dr),
    }


def stage_min_steps(stage, args):
    return {
        0: int(args.stage0_min_steps),
        1: int(args.stage1_min_steps),
        2: int(args.stage2_min_steps),
        3: 10 ** 18,
    }[int(stage)]


def _weighted_cycle(cycle_index):
    cycle = list(MULTIMAP_WEIGHTED_SLOTS)
    rng = random.Random(91000 + int(cycle_index) * 7919)
    rng.shuffle(cycle)
    return cycle


def assigned_map(stage, stage_block_index, worker_id, worker_count):
    stage = int(stage)
    block = int(stage_block_index)
    worker_id = int(worker_id)
    worker_count = int(worker_count)

    if stage <= 1:
        # With X2, every block covers both anchor maps and swaps worker ownership.
        return ANCHOR_MAPS[(block + worker_id) % len(ANCHOR_MAPS)]

    global_slot = block * worker_count + worker_id
    cycle_len = len(MULTIMAP_WEIGHTED_SLOTS)
    cycle_index = global_slot // cycle_len
    position = global_slot % cycle_len
    return _weighted_cycle(cycle_index)[position]


# ============================================================================
# State / checkpoint / champion
# ============================================================================

def checkpoint_dir():
    path = os.path.join(PPO_CHECKPOINT_DIR, MODEL_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def state_path():
    return os.path.join(checkpoint_dir(), STATE_NAME)


def champion_path():
    return os.path.join(checkpoint_dir(), CHAMPION_NAME)


def numbered_policy_files():
    pattern = re.compile(r"^ppo_policy_(\d+)_\.pth$")
    items = []
    for name in os.listdir(checkpoint_dir()):
        m = pattern.match(name)
        if m:
            items.append((int(m.group(1)), os.path.join(checkpoint_dir(), name)))
    items.sort(key=lambda x: x[0])
    return items


def read_state():
    with open(state_path(), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("version") != TRAINER_VERSION:
        raise RuntimeError("Training state version mismatch: {}".format(data.get("version")))
    return data


def write_state(agent, global_step, episode, stage, stage_start_step,
                consecutive_passes, checkpoint_path_value, args):
    """
    Resume state always points to the last VERIFIED main checkpoint.

    GUARDED V2 rule:
      - numbered ppo_policy_N_.pth => CANARY PASS only
      - rejected/                  => diagnostic only
      - emergency/                 => unverified interruption recovery only
    """
    data = {
        "version": TRAINER_VERSION,
        "model_name": MODEL_NAME,
        "checkpoint_gate": "CANARY_PASS_ONLY",
        "global_step": int(global_step),
        "episode": int(episode),
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "stage_start_step": int(stage_start_step),
        "consecutive_canary_passes": int(consecutive_passes),
        "action_std": float(agent.action_std),
        "checkpoint_path": str(checkpoint_path_value),
        "champion_path": champion_path() if os.path.isfile(champion_path()) else None,
        "safe_spawns": list(args.safe_spawns),
        "map_block_steps": int(args.map_block_steps),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(state_path(), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return state_path()


def _v5_clean_payload(agent):
    """Create a PPOAgent-V5-loadable checkpoint payload for non-main snapshots."""
    return {
        "version": "PPO_V5_CLEAN",
        "obs_dim": int(agent.obs_dim),
        "action_dim": int(agent.action_dim),
        "action_std": float(agent.action_std),
        "policy_state_dict": copy.deepcopy(agent.old_policy.state_dict()),
        "optimizer_state_dict": copy.deepcopy(agent.optimizer.state_dict()),
    }


def _prune_snapshot_dir(directory, prefix, keep):
    keep = int(keep)
    if not os.path.isdir(directory):
        return
    items = []
    for name in os.listdir(directory):
        if not name.startswith(prefix) or not name.endswith(".pth"):
            continue
        full = os.path.join(directory, name)
        try:
            stamp = os.path.getmtime(full)
        except Exception:
            stamp = 0.0
        items.append((float(stamp), name, full))
    items.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, full in items[max(keep, 0):]:
        try:
            os.remove(full)
        except Exception:
            logging.exception("Could not prune snapshot %s", full)


def save_passed_checkpoint(agent, global_step, episode, stage, stage_start_step,
                           consecutive_passes, canary_summary, args):
    """The ONLY function allowed to create a main numbered PPO checkpoint."""
    path = agent.save()

    sidecar = path + ".canary.json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump({
            "gate": "CANARY_PASS",
            "global_step": int(global_step),
            "episode": int(episode),
            "stage": int(stage),
            "stage_name": STAGE_NAMES[int(stage)],
            "canary_summary": dict(canary_summary),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }, handle, indent=2, sort_keys=True)

    # State is written only AFTER champion save in main(), so the state file
    # always points at an already-existing champion + verified policy pair.
    print(
        "PASSED POLICY FILE | step={} | stage={} | policy={} | canary={}".format(
            global_step, STAGE_NAMES[int(stage)], path, sidecar
        )
    )
    return path


def save_champion(agent, global_step, stage, canary_summary, checkpoint_path_value):
    payload = {
        "version": "PPO_V5_MULTIMAP_CHAMPION_2",
        "global_step": int(global_step),
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "action_std": float(agent.action_std),
        "checkpoint_path": str(checkpoint_path_value),
        "policy_state_dict": copy.deepcopy(agent.policy.state_dict()),
        "old_policy_state_dict": copy.deepcopy(agent.old_policy.state_dict()),
        "optimizer_state_dict": copy.deepcopy(agent.optimizer.state_dict()),
        "canary_summary": dict(canary_summary),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, champion_path())
    return champion_path()


def load_champion(agent):
    path = champion_path()
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location=agent.device)
    if payload.get("version") != "PPO_V5_MULTIMAP_CHAMPION_2":
        raise RuntimeError("Unknown champion version: {}".format(payload.get("version")))
    agent.policy.load_state_dict(payload["policy_state_dict"], strict=True)
    agent.old_policy.load_state_dict(payload["old_policy_state_dict"], strict=True)
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.set_action_std(float(payload["action_std"]))
    agent.memory.clear()
    return payload


def save_rejected_snapshot(agent, global_step, stage, reasons, summary, keep=5):
    """Save FAIL for diagnosis only. This file is NOT a main checkpoint."""
    directory = os.path.join(checkpoint_dir(), "rejected")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(
        directory,
        "rejected_step{}_stage{}_{}.pth".format(
            int(global_step), int(stage), datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
    payload = _v5_clean_payload(agent)
    payload["guarded_snapshot_kind"] = "CANARY_REJECTED"
    payload["guarded_metadata"] = {
        "global_step": int(global_step),
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "reasons": list(reasons),
        "canary_summary": dict(summary),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, path)
    _prune_snapshot_dir(directory, "rejected_", int(keep))
    return path


def save_emergency_snapshot(agent, global_step, episode, stage, stage_start_step,
                            consecutive_passes, args, reason):
    """
    Save an UNVERIFIED but PPO_V5_CLEAN-loadable snapshot outside the main
    checkpoint namespace. It can later be used explicitly with --init-checkpoint.
    """
    directory = os.path.join(checkpoint_dir(), "emergency")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(
        directory,
        "emergency_step{}_stage{}_{}.pth".format(
            int(global_step), int(stage), datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
    )
    payload = _v5_clean_payload(agent)
    payload["guarded_snapshot_kind"] = "EMERGENCY_UNVERIFIED"
    payload["guarded_metadata"] = {
        "reason": str(reason),
        "global_step": int(global_step),
        "episode": int(episode),
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "stage_start_step": int(stage_start_step),
        "consecutive_canary_passes": int(consecutive_passes),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, path)
    _prune_snapshot_dir(directory, "emergency_", int(args.keep_emergency))
    return path


def prepare_fresh_or_resume(agent, args):
    files = numbered_policy_files()
    state_exists = os.path.isfile(state_path())

    if not bool(args.resume):
        if files or state_exists or os.path.isfile(champion_path()):
            raise RuntimeError(
                "FRESH SAFETY STOP: '{}' already contains training state/checkpoints. "
                "Archive it or use --resume true.".format(checkpoint_dir())
            )
        init_checkpoint = str(args.init_checkpoint).strip()
        if init_checkpoint:
            if not os.path.isfile(init_checkpoint):
                raise FileNotFoundError(init_checkpoint)
            loaded = agent.load(checkpoint_path=init_checkpoint)
            agent.memory.clear()
            print("FRESH MULTIMAP | initialized from {} | global_step=0".format(loaded))
            return 0, 0, 0, 0, 0, loaded
        print("FRESH MULTIMAP | random init | global_step=0")
        return 0, 0, 0, 0, 0, None

    if not state_exists:
        raise FileNotFoundError("Missing resume state: {}".format(state_path()))
    state = read_state()
    cp = str(state.get("checkpoint_path", ""))
    if not cp or not os.path.isfile(cp):
        raise FileNotFoundError("Resume checkpoint missing: {}".format(cp))
    loaded = agent.load(checkpoint_path=cp)
    # State may be newer than the champion checkpoint after a rollback.
    # Preserve the exploration schedule at the recorded global_step.
    if "action_std" in state:
        agent.set_action_std(float(state["action_std"]))
    agent.memory.clear()
    return (
        int(state.get("global_step", 0)),
        int(state.get("episode", 0)),
        int(state.get("stage", 0)),
        int(state.get("stage_start_step", 0)),
        int(state.get("consecutive_canary_passes", 0)),
        loaded,
    )


# ============================================================================
# Worker runtime
# ============================================================================

class WorkerRuntime(object):
    def __init__(self, worker_id, cfg):
        self.worker_id = int(worker_id)
        self.cfg = dict(cfg)
        torch.set_num_threads(int(cfg["worker_torch_threads"]))
        seed_everything(int(cfg["seed"]))

        self.client = carla.Client(str(cfg["host"]), int(cfg["port"]))
        self.client.set_timeout(float(cfg["carla_timeout"]))

        self.agent = PPOAgent(
            town="__multimap_guard_worker_{}".format(worker_id),
            action_std_init=float(cfg["action_std_init"]),
            device=cfg["worker_ppo_device"],
        )

        self.env = None
        self.observation = None
        self.current_map = None
        self.current_signature = None
        self.episode_stats = EpisodeAccumulator()
        self.completed_episode_count = 0

    def close_env(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        self.observation = None
        self.current_map = None
        self.current_signature = None
        self.episode_stats = EpisodeAccumulator()

    def close(self):
        self.close_env()

    def _load_world(self, map_path):
        world = self.client.load_world(str(map_path))
        actual = str(world.get_map().name).replace("\\", "/").lower()
        expected = str(map_path).replace("\\", "/").lower()
        tail = expected.split("/")[-1]
        if tail and tail not in actual:
            raise RuntimeError(
                "Worker {} map mismatch: requested='{}' actual='{}'".format(
                    self.worker_id, map_path, world.get_map().name
                )
            )
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        return world

    def _configure_recovery(self, command):
        configure_recovery_scenarios_v5(
            recovery_spawn_probability=0.0,
            spawn_lateral_min_m=0.0,
            spawn_lateral_max_m=0.0,
            spawn_heading_min_deg=0.0,
            spawn_heading_max_deg=0.0,
            disturbance_probability=float(command["recovery_prob"]),
            disturbance_duration_min_s=float(self.cfg["disturbance_duration_min_s"]),
            disturbance_duration_max_s=float(self.cfg["disturbance_duration_max_s"]),
            disturbance_steer_min=float(self.cfg["disturbance_steer_min"]),
            disturbance_steer_max=float(self.cfg["disturbance_steer_max"]),
            recovery_min_episode_step=int(self.cfg["recovery_min_episode_step"]),
            recovery_stable_ticks=int(self.cfg["recovery_stable_ticks"]),
            recovery_stable_lateral_m=float(self.cfg["recovery_stable_lateral_m"]),
            recovery_stable_heading_deg=float(self.cfg["recovery_stable_heading_deg"]),
            recovery_stable_speed_mps=float(self.cfg["recovery_stable_speed_mps"]),
            recovery_cooldown_ticks=int(self.cfg["recovery_cooldown_ticks"]),
            recovery_max_events_per_episode=int(self.cfg["recovery_max_events"]),
            seed=int(command["scenario_seed"]),
        )

    def ensure_train_env(self, command):
        signature = (
            str(command["map_path"]),
            bool(command["dynamics_dr"]),
            bool(command["vision_dr"]),
            bool(command["sensor_dr"]),
            bool(command["recovery"]),
            round(float(command["recovery_prob"]), 6),
            tuple(int(x) for x in command["safe_spawns"]),
            int(command["scenario_seed"]),
        )

        if self.env is not None and signature == self.current_signature:
            return

        self.close_env()
        self._configure_recovery(command)
        world = self._load_world(command["map_path"])

        self.env = CarlaEnvironmentRGBV5MultiMap(
            client=self.client,
            world=world,
            town="worker_{}".format(self.worker_id),
            safe_spawn_numbers=list(command["safe_spawns"]),
            desired_speed_mps=float(command["desired_speed"]),
            max_episode_seconds=float(self.cfg["max_episode_seconds"]),
            encoder_device=self.cfg["encoder_device"],
            dynamics_dr_enabled=bool(command["dynamics_dr"]),
            vision_dr_enabled=bool(command["vision_dr"]),
            sensor_dr_enabled=bool(command["sensor_dr"]),
            domain_randomization_seed=int(command["dr_seed"]),
            recovery_scenarios_enabled=bool(command["recovery"]),
        )
        self.current_map = str(command["map_path"])
        self.current_signature = signature
        self.observation = self.env.reset()
        self.episode_stats = EpisodeAccumulator()

    def collect(self, command):
        self.ensure_train_env(command)

        load_numpy_state(self.agent.policy, command["policy_state"])
        load_numpy_state(self.agent.old_policy, command["policy_state"])
        self.agent.set_action_std(float(command["action_std"]))
        self.agent.memory.clear()
        self.env.set_desired_speed_mps(float(command["desired_speed"]))

        requested_steps = int(command["steps"])
        diagnostics = RolloutDiagnostics()
        completed = []
        start = time.time()
        artificial_cut = False

        for local_i in range(requested_steps):
            action = self.agent.get_action(self.observation, train=True)
            next_obs, reward, done, info = self.env.step(action)

            actual_terminated = bool(info.get("terminated", False))
            actual_truncated = bool(info.get("truncated", False))
            is_last = local_i == requested_steps - 1
            artificial_cut = bool(is_last and not actual_terminated and not actual_truncated)
            train_truncated = bool(actual_truncated or artificial_cut)

            self.agent.record_outcome(
                reward=reward,
                terminated=actual_terminated,
                truncated=train_truncated,
                next_obs=(next_obs if train_truncated else None),
            )

            self.episode_stats.add(action=action, reward=reward, info=info)
            diagnostics.add(reward=reward, info=info)

            if done:
                self.completed_episode_count += 1
                summary = self.episode_stats.finish(info)
                summary["map_path"] = self.current_map
                summary["spawn_number"] = info.get("spawn_number")
                completed.append(summary)
                self.observation = self.env.reset()
            else:
                self.observation = next_obs

        rollout = export_memory(self.agent.memory)
        self.agent.memory.clear()

        return {
            "type": "rollout",
            "worker": self.worker_id,
            "map_path": self.current_map,
            "samples": requested_steps,
            "wall_s": float(time.time() - start),
            "rollout": rollout,
            "diagnostics": diagnostics.export(),
            "episodes": completed,
            "worker_episode_count": self.completed_episode_count,
            "artificial_final_cut": bool(artificial_cut),
        }

    def evaluate_anchor(self, command):
        # Canary intentionally destroys the current training env and uses a
        # completely clean deterministic environment.
        self.close_env()

        eval_command = {
            "recovery_prob": 0.0,
            "scenario_seed": int(command["scenario_seed"]),
        }
        self._configure_recovery(eval_command)
        world = self._load_world(command["map_path"])

        env = CarlaEnvironmentRGBV5MultiMap(
            client=self.client,
            world=world,
            town="__canary_worker_{}".format(self.worker_id),
            safe_spawn_numbers=list(command["safe_spawns"]),
            desired_speed_mps=1.0,
            max_episode_seconds=(float(command["max_steps"]) / CONTROL_HZ + 5.0),
            encoder_device=self.cfg["encoder_device"],
            dynamics_dr_enabled=False,
            vision_dr_enabled=False,
            sensor_dr_enabled=False,
            domain_randomization_seed=int(command["dr_seed"]),
            recovery_scenarios_enabled=False,
        )

        load_numpy_state(self.agent.policy, command["policy_state"])
        load_numpy_state(self.agent.old_policy, command["policy_state"])
        self.agent.memory.clear()

        results = []
        try:
            for spawn_number in command["safe_spawns"]:
                env.force_next_spawn_number(int(spawn_number))
                obs = env.reset()

                n = 0
                reward_sum = 0.0
                progress_sum = 0.0
                abs_lat_sum = 0.0
                max_abs_lat = 0.0
                abs_heading_sum = 0.0
                max_abs_heading = 0.0
                abs_steer_sum = 0.0
                steer_gt_050 = 0
                steer_gt_075 = 0
                steer_pos = 0
                steer_neg = 0
                steer_neutral = 0
                edge_ticks = 0
                center_escape_count = 0
                stable_ticks = 0
                stable_large_steer = 0
                yaw_pos_sum = 0.0
                yaw_pos_count = 0
                yaw_neg_sum = 0.0
                yaw_neg_count = 0
                initial_rho = None
                done = False
                reason = None

                for step_idx in range(int(command["max_steps"])):
                    action = self.agent.get_action(obs, train=False)
                    next_obs, reward, done, info = env.step(action)
                    n += 1
                    reward_sum += float(reward)
                    progress_sum += float(info.get("forward_progress_m", 0.0))

                    lat = abs(float(info.get("lateral_error_m", 0.0)))
                    heading = abs(float(info.get("heading_error_rad", 0.0)))
                    abs_lat_sum += lat
                    max_abs_lat = max(max_abs_lat, lat)
                    abs_heading_sum += heading
                    max_abs_heading = max(max_abs_heading, heading)

                    control = info.get("control", {})
                    if isinstance(control, dict):
                        steer = float(control.get("actual_steer_cmd", action[0]))
                    else:
                        steer = float(action[0])
                    abs_steer = abs(steer)
                    abs_steer_sum += abs_steer
                    if abs_steer >= 0.50:
                        steer_gt_050 += 1
                    if abs_steer >= 0.75:
                        steer_gt_075 += 1
                    if steer > 0.05:
                        steer_pos += 1
                    elif steer < -0.05:
                        steer_neg += 1
                    else:
                        steer_neutral += 1

                    imu = info.get("imu_clean", {})
                    yaw = float(imu.get("yaw_rate_rad_s", 0.0)) if isinstance(imu, dict) else 0.0
                    if steer > 0.20:
                        yaw_pos_sum += yaw
                        yaw_pos_count += 1
                    elif steer < -0.20:
                        yaw_neg_sum += yaw
                        yaw_neg_count += 1

                    half_lane_real = None
                    try:
                        wp = env.reward_manager.road_metrics._driving_waypoint_projected()
                        if wp is not None:
                            half_lane_real = (
                                float(wp.lane_width)
                                / float(CARLA_GEOMETRY_SCALE)
                                / 2.0
                            )
                    except Exception:
                        half_lane_real = None

                    if half_lane_real is not None and half_lane_real > 1e-6:
                        rho = lat / half_lane_real
                        if initial_rho is None:
                            initial_rho = float(rho)
                        if rho >= float(command["edge_ratio_threshold"]):
                            edge_ticks += 1
                        if (
                            step_idx < int(command["center_escape_window_ticks"])
                            and initial_rho <= float(command["center_start_ratio"])
                            and rho >= float(command["center_escape_ratio"])
                        ):
                            center_escape_count = 1

                    speed = float(info.get("speed_mps", 0.0))
                    stable = bool(
                        lat <= 0.025
                        and heading <= math.radians(5.0)
                        and speed >= 0.30
                    )
                    if stable:
                        stable_ticks += 1
                        if abs_steer >= 0.50:
                            stable_large_steer += 1

                    obs = next_obs
                    if done:
                        reason = info.get("termination_reason")
                        break

                denom = float(max(n, 1))
                results.append({
                    "worker": self.worker_id,
                    "map_path": str(command["map_path"]),
                    "spawn_number": int(spawn_number),
                    "steps": int(n),
                    "reason": reason if done else "canary_horizon",
                    "terminated": bool(done),
                    "reward": float(reward_sum),
                    "progress_m": float(progress_sum),
                    "avg_abs_lateral_m": float(abs_lat_sum / denom),
                    "max_abs_lateral_m": float(max_abs_lat),
                    "avg_abs_heading_deg": float(math.degrees(abs_heading_sum / denom)),
                    "max_abs_heading_deg": float(math.degrees(max_abs_heading)),
                    "avg_abs_steer": float(abs_steer_sum / denom),
                    "steer_gt_050_rate": float(steer_gt_050 / denom),
                    "steer_gt_075_rate": float(steer_gt_075 / denom),
                    "steer_positive_rate": float(steer_pos / denom),
                    "steer_negative_rate": float(steer_neg / denom),
                    "steer_neutral_rate": float(steer_neutral / denom),
                    "edge_occupancy_rate": float(edge_ticks / denom),
                    "center_escape_count": int(center_escape_count),
                    "stable_large_steer_rate": (
                        float(stable_large_steer) / float(stable_ticks)
                        if stable_ticks > 0 else 0.0
                    ),
                    "mean_yaw_when_steer_positive": (
                        float(yaw_pos_sum) / float(yaw_pos_count)
                        if yaw_pos_count > 0 else 0.0
                    ),
                    "mean_yaw_when_steer_negative": (
                        float(yaw_neg_sum) / float(yaw_neg_count)
                        if yaw_neg_count > 0 else 0.0
                    ),
                })
        finally:
            try:
                env.close()
            except Exception:
                pass
            self.env = None
            self.observation = None
            self.current_map = None
            self.current_signature = None

        return {
            "type": "canary",
            "worker": self.worker_id,
            "map_path": str(command["map_path"]),
            "results": results,
        }


def worker_main(worker_id, conn, cfg):
    runtime = None
    try:
        runtime = WorkerRuntime(worker_id, cfg)
        current_world = runtime.client.get_world()
        conn.send({
            "type": "ready",
            "worker": int(worker_id),
            "port": int(cfg["port"]),
            "current_map": str(current_world.get_map().name),
        })

        while True:
            command = conn.recv()
            kind = command.get("cmd")
            if kind == "close":
                break
            if kind == "collect":
                conn.send(runtime.collect(command))
            elif kind == "evaluate_anchor":
                conn.send(runtime.evaluate_anchor(command))
            else:
                raise RuntimeError("Unknown worker command: {}".format(kind))

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            conn.send({
                "type": "error",
                "worker": int(worker_id),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        if runtime is not None:
            runtime.close()
        try:
            conn.close()
        except Exception:
            pass


# ============================================================================
# Canary evaluation / policy guard
# ============================================================================

def run_canary(parent_conns, master, args, global_step):
    snapshot = state_dict_to_numpy(master.old_policy)
    active = []

    if int(args.workers) == 1:
        assignments = [(0, ANCHOR_MAPS[0]), (0, ANCHOR_MAPS[1])]
        all_results = []
        for worker_id, map_path in assignments:
            command = {
                "cmd": "evaluate_anchor",
                "map_path": map_path,
                "safe_spawns": list(args.safe_spawns),
                "max_steps": int(args.canary_max_steps),
                "policy_state": snapshot,
                "scenario_seed": 800000 + int(global_step) + worker_id,
                "dr_seed": 900000 + int(global_step) + worker_id,
                "edge_ratio_threshold": float(args.canary_edge_ratio_threshold),
                "center_start_ratio": float(args.canary_center_start_ratio),
                "center_escape_ratio": float(args.canary_center_escape_ratio),
                "center_escape_window_ticks": int(args.canary_center_escape_window_ticks),
            }
            parent_conns[0].send(command)
            if not parent_conns[0].poll(float(args.worker_timeout)):
                raise TimeoutError("Canary worker timeout.")
            msg = parent_conns[0].recv()
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("traceback", msg.get("error")))
            all_results.extend(msg["results"])
        return all_results

    for worker_id, map_path in enumerate(ANCHOR_MAPS):
        command = {
            "cmd": "evaluate_anchor",
            "map_path": map_path,
            "safe_spawns": list(args.safe_spawns),
            "max_steps": int(args.canary_max_steps),
            "policy_state": snapshot,
            "scenario_seed": 800000 + int(global_step) + worker_id,
            "dr_seed": 900000 + int(global_step) + worker_id,
            "edge_ratio_threshold": float(args.canary_edge_ratio_threshold),
            "center_start_ratio": float(args.canary_center_start_ratio),
            "center_escape_ratio": float(args.canary_center_escape_ratio),
            "center_escape_window_ticks": int(args.canary_center_escape_window_ticks),
        }
        parent_conns[worker_id].send(command)
        active.append(worker_id)

    all_results = []
    for worker_id in active:
        conn = parent_conns[worker_id]
        if not conn.poll(float(args.worker_timeout)):
            raise TimeoutError("Canary worker {} timeout.".format(worker_id))
        msg = conn.recv()
        if msg.get("type") == "error":
            raise RuntimeError(msg.get("traceback", msg.get("error")))
        if msg.get("type") != "canary":
            raise RuntimeError("Unexpected canary message: {}".format(msg))
        all_results.extend(msg["results"])

    return all_results


def canary_summary(results):
    if not results:
        return {
            "cases": 0,
            "avg_abs_lateral_m": 999.0,
            "max_abs_lateral_m": 999.0,
            "avg_abs_heading_deg": 999.0,
            "avg_abs_steer": 999.0,
            "steer_gt_075_rate": 1.0,
            "edge_occupancy_rate": 1.0,
            "center_escape_count": 999,
            "progress_m_mean": 0.0,
            "steer_positive_rate": 0.0,
            "steer_negative_rate": 0.0,
        }

    def mean(key):
        return float(np.mean([float(x.get(key, 0.0)) for x in results]))

    return {
        "cases": int(len(results)),
        "avg_abs_lateral_m": mean("avg_abs_lateral_m"),
        "max_abs_lateral_m": float(max(float(x["max_abs_lateral_m"]) for x in results)),
        "avg_abs_heading_deg": mean("avg_abs_heading_deg"),
        "avg_abs_steer": mean("avg_abs_steer"),
        "steer_gt_075_rate": mean("steer_gt_075_rate"),
        "edge_occupancy_rate": mean("edge_occupancy_rate"),
        "center_escape_count": int(sum(int(x["center_escape_count"]) for x in results)),
        "progress_m_mean": mean("progress_m"),
        "steer_positive_rate": mean("steer_positive_rate"),
        "steer_negative_rate": mean("steer_negative_rate"),
        "stable_large_steer_rate": mean("stable_large_steer_rate"),
        "mean_yaw_when_steer_positive": mean("mean_yaw_when_steer_positive"),
        "mean_yaw_when_steer_negative": mean("mean_yaw_when_steer_negative"),
    }


def judge_canary(results, args, champion_payload=None):
    reasons = []

    expected_cases = len(ANCHOR_MAPS) * len(args.safe_spawns)
    if len(results) != expected_cases:
        reasons.append("expected {} canary cases, got {}".format(expected_cases, len(results)))

    for item in results:
        label = "{} spawn{}".format(item["map_path"].split("/")[-1], item["spawn_number"])
        reason = str(item.get("reason"))
        if reason in ("offroad", "collision", "stuck"):
            reasons.append("{} terminated={}".format(label, reason))
        if float(item["avg_abs_lateral_m"]) > float(args.canary_avg_lateral_max_m):
            reasons.append("{} avg_lat {:.4f}>{:.4f}".format(
                label, item["avg_abs_lateral_m"], args.canary_avg_lateral_max_m
            ))
        if float(item["max_abs_lateral_m"]) > float(args.canary_max_lateral_max_m):
            reasons.append("{} max_lat {:.4f}>{:.4f}".format(
                label, item["max_abs_lateral_m"], args.canary_max_lateral_max_m
            ))
        if float(item["avg_abs_heading_deg"]) > float(args.canary_avg_heading_max_deg):
            reasons.append("{} avg_heading {:.2f}>{:.2f}".format(
                label, item["avg_abs_heading_deg"], args.canary_avg_heading_max_deg
            ))
        if float(item["edge_occupancy_rate"]) > float(args.canary_edge_rate_max):
            reasons.append("{} edge_rate {:.1f}%>{:.1f}%".format(
                label, 100.0 * item["edge_occupancy_rate"], 100.0 * args.canary_edge_rate_max
            ))
        if float(item["steer_gt_075_rate"]) > float(args.canary_steer075_rate_max):
            reasons.append("{} steer>.75 {:.1f}%>{:.1f}%".format(
                label, 100.0 * item["steer_gt_075_rate"], 100.0 * args.canary_steer075_rate_max
            ))
        if float(item["progress_m"]) < float(args.canary_min_progress_m):
            reasons.append("{} progress {:.2f}<{:.2f}m".format(
                label, item["progress_m"], args.canary_min_progress_m
            ))
        if int(item["center_escape_count"]) > 0:
            reasons.append("{} CENTER_ESCAPE".format(label))

    summary = canary_summary(results)

    if champion_payload is not None:
        champ = champion_payload.get("canary_summary", {})
        champ_lat = float(champ.get("avg_abs_lateral_m", 0.0))
        if champ_lat > 0.0:
            allowed = (
                champ_lat * float(args.canary_regression_factor)
                + float(args.canary_regression_slack_m)
            )
            if float(summary["avg_abs_lateral_m"]) > allowed:
                reasons.append(
                    "anchor avg_lat regression {:.4f}>{:.4f} vs champion {:.4f}".format(
                        summary["avg_abs_lateral_m"], allowed, champ_lat
                    )
                )

    return (len(reasons) == 0), reasons, summary


def print_canary(results, passed, reasons, summary, global_step, stage):
    print("\n" + "=" * 118)
    print("CANARY | step={} | stage={} | {}".format(
        global_step, STAGE_NAMES[int(stage)], "PASS" if passed else "FAIL"
    ))
    print("=" * 118)
    for item in results:
        print(
            "  {:12s} spawn{} | steps={:4d} reason={:14s} | progress={:6.2f}m | "
            "lat={:.4f}/{:.4f}m | head={:.2f}deg | |steer|={:.3f} >.75={:4.1f}% | "
            "edge={:4.1f}% escape={} | +={:4.1f}% -={:4.1f}%".format(
                item["map_path"].split("/")[-1],
                item["spawn_number"],
                item["steps"],
                str(item["reason"]),
                item["progress_m"],
                item["avg_abs_lateral_m"],
                item["max_abs_lateral_m"],
                item["avg_abs_heading_deg"],
                item["avg_abs_steer"],
                100.0 * item["steer_gt_075_rate"],
                100.0 * item["edge_occupancy_rate"],
                item["center_escape_count"],
                100.0 * item["steer_positive_rate"],
                100.0 * item["steer_negative_rate"],
            )
        )
    print(
        "  SUMMARY | lat={:.4f}m max={:.4f}m | head={:.2f}deg | |steer|={:.3f} | "
        "edge={:.1f}% | escapes={} | steer +={:.1f}% -={:.1f}% | "
        "yaw(+steer)={:+.4f} yaw(-steer)={:+.4f}".format(
            summary["avg_abs_lateral_m"],
            summary["max_abs_lateral_m"],
            summary["avg_abs_heading_deg"],
            summary["avg_abs_steer"],
            100.0 * summary["edge_occupancy_rate"],
            summary["center_escape_count"],
            100.0 * summary["steer_positive_rate"],
            100.0 * summary["steer_negative_rate"],
            summary["mean_yaw_when_steer_positive"],
            summary["mean_yaw_when_steer_negative"],
        )
    )
    if reasons:
        for reason in reasons:
            print("  FAIL REASON | {}".format(reason))
    print("=" * 118)


# ============================================================================
# Main
# ============================================================================

def maybe_decay_action_std(agent, global_step, previous_bucket, args):
    bucket = int(global_step // int(args.action_std_decay_freq))
    if bucket <= previous_bucket:
        return previous_bucket
    for _ in range(bucket - previous_bucket):
        if agent.action_std <= float(args.action_std_min) + 1e-12:
            break
        agent.decay_action_std(
            action_std_decay_rate=float(args.action_std_decay),
            min_action_std=float(args.action_std_min),
        )
    print("ACTION STD | step={} | std={:.4f}".format(global_step, agent.action_std))
    return bucket


def make_writer(args):
    if not bool(args.tensorboard) or SummaryWriter is None:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "{}_{}".format(MODEL_NAME, stamp))
    writer = SummaryWriter(run_dir)
    writer.add_text("config", "\n".join(
        "{} = {}".format(k, v) for k, v in sorted(vars(args).items())
    ))
    print("TensorBoard:", run_dir)
    return writer


def worker_configs(args):
    sources = [(args.port0, args.seed0), (args.port1, args.seed1)]
    result = []
    for worker_id in range(int(args.workers)):
        port, seed = sources[worker_id]
        result.append({
            "host": args.host,
            "port": int(port),
            "seed": int(seed),
            "carla_timeout": float(args.carla_timeout),
            "max_episode_seconds": float(args.max_episode_seconds),
            "encoder_device": args.encoder_device,
            "worker_ppo_device": args.worker_ppo_device,
            "worker_torch_threads": int(args.worker_torch_threads),
            "action_std_init": float(args.action_std_init),
            "disturbance_duration_min_s": float(args.disturbance_duration_min_s),
            "disturbance_duration_max_s": float(args.disturbance_duration_max_s),
            "disturbance_steer_min": float(args.disturbance_steer_min),
            "disturbance_steer_max": float(args.disturbance_steer_max),
            "recovery_min_episode_step": int(args.recovery_min_episode_step),
            "recovery_stable_ticks": int(args.recovery_stable_ticks),
            "recovery_stable_lateral_m": float(args.recovery_stable_lateral_m),
            "recovery_stable_heading_deg": float(args.recovery_stable_heading_deg),
            "recovery_stable_speed_mps": float(args.recovery_stable_speed_mps),
            "recovery_cooldown_ticks": int(args.recovery_cooldown_ticks),
            "recovery_max_events": int(args.recovery_max_events),
        })
    return result


def main():
    args = parse_args()
    validate_args(args)
    mp.freeze_support()
    torch.set_num_threads(int(args.learner_torch_threads))
    seed_everything(int(SEED))

    master = PPOAgent(
        town=MODEL_NAME,
        action_std_init=float(args.action_std_init),
        device=args.learner_device,
    )

    (
        global_step,
        episode,
        stage,
        stage_start_step,
        consecutive_passes,
        loaded_checkpoint,
    ) = prepare_fresh_or_resume(master, args)

    target_step = int(args.total_timesteps)
    writer = make_writer(args)
    cfgs = worker_configs(args)

    ctx = mp.get_context("spawn")
    parent_conns = []
    processes = []
    run_start = time.time()

    # Existing champion is loaded only as rollback reference. Current policy
    # remains the verified resume checkpoint from training state.
    champion_payload = None
    last_valid_checkpoint_path = (
        str(loaded_checkpoint) if bool(args.resume) and loaded_checkpoint else None
    )
    if os.path.isfile(champion_path()):
        champion_payload = torch.load(champion_path(), map_location=master.device)
        if champion_payload.get("version") != "PPO_V5_MULTIMAP_CHAMPION_2":
            raise RuntimeError(
                "Champion version mismatch: {}".format(champion_payload.get("version"))
            )
        champion_cp = str(champion_payload.get("checkpoint_path", ""))
        if champion_cp and os.path.isfile(champion_cp):
            last_valid_checkpoint_path = champion_cp

    try:
        for worker_id in range(int(args.workers)):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            process = ctx.Process(
                target=worker_main,
                args=(worker_id, child_conn, cfgs[worker_id]),
                name="CARLA-V5-MULTIMAP-W{}".format(worker_id),
            )
            process.daemon = False
            process.start()
            child_conn.close()
            parent_conns.append(parent_conn)
            processes.append(process)

        ready = []
        for worker_id, conn in enumerate(parent_conns):
            if not conn.poll(float(args.worker_timeout)):
                raise TimeoutError("Worker {} init timeout.".format(worker_id))
            msg = conn.recv()
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("traceback", msg.get("error")))
            ready.append(msg)

        print("=" * 118)
        print("PPO V5 MULTI-OVAL GUARDED TRAINER")
        print("=" * 118)
        print("model             :", MODEL_NAME)
        print("loaded            :", loaded_checkpoint if loaded_checkpoint else "RANDOM INIT")
        print("global step       :", global_step)
        print("target step       :", target_step)
        print("stage             : {} ({})".format(stage, STAGE_NAMES[stage]))
        print("safe spawns       :", list(args.safe_spawns), "| CLEAN, no spawn perturbation")
        print("anchor maps       :", ANCHOR_MAPS)
        print("aux oval maps     :", AUX_OVAL_MAPS)
        print("left-only stress  :", LEFT_ONLY_STRESS_MAP)
        print("multimap weights  : anchor=58.8% | aux=35.3% | left-only=5.9%")
        print("episode mix       : nominal={:.0f}% | delayed recovery={:.0f}% in recovery stages".format(
            100.0 * (1.0 - float(args.recovery_episode_prob)),
            100.0 * float(args.recovery_episode_prob),
        ))
        print("canary            : every {} steps | {} ticks/spawn | anchors x {} spawns".format(
            args.canary_every_steps, args.canary_max_steps, len(args.safe_spawns)
        ))
        for msg in ready:
            print("worker {}          : port={} current_map={}".format(
                msg["worker"], msg["port"], msg["current_map"]
            ))
        print("=" * 118)

        if args.smoke_test:
            target_step = global_step + int(args.smoke_steps_per_worker) * int(args.workers)

        print(
            "checkpoint gate    : CANARY PASS ONLY | --checkpoint-every-steps is ignored"
        )
        print(
            "snapshot folders  : rejected/ keep={} | emergency/ keep={}".format(
                int(args.keep_rejected), int(args.keep_emergency)
            )
        )

        next_canary_step = (
            (global_step // int(args.canary_every_steps)) + 1
        ) * int(args.canary_every_steps)
        decay_bucket = int(global_step // int(args.action_std_decay_freq))
        update_index = 0

        while global_step < target_step:
            update_index += 1
            requested_total = min(int(args.rollout_total), int(target_step - global_step))
            per_worker = split_steps(requested_total, int(args.workers))

            stage_cfg = stage_config(stage, args)
            stage_block_index = int(
                max(0, global_step - stage_start_step) // int(args.map_block_steps)
            )
            policy_snapshot = state_dict_to_numpy(master.old_policy)

            active = []
            collect_start = time.time()
            for worker_id, steps in enumerate(per_worker):
                if steps <= 0:
                    continue
                map_path = assigned_map(
                    stage=stage,
                    stage_block_index=stage_block_index,
                    worker_id=worker_id,
                    worker_count=int(args.workers),
                )
                scenario_seed = (
                    int(cfgs[worker_id]["seed"])
                    + int(stage) * 100000
                    + int(stage_block_index) * 1000
                )
                dr_seed = scenario_seed + 60000
                parent_conns[worker_id].send({
                    "cmd": "collect",
                    "steps": int(steps),
                    "policy_state": policy_snapshot,
                    "action_std": float(master.action_std),
                    "map_path": map_path,
                    "safe_spawns": list(args.safe_spawns),
                    "desired_speed": float(args.desired_speed),
                    "recovery": bool(stage_cfg["recovery"]),
                    "recovery_prob": float(stage_cfg["recovery_prob"]),
                    "dynamics_dr": bool(stage_cfg["dynamics_dr"]),
                    "vision_dr": bool(stage_cfg["vision_dr"]),
                    "sensor_dr": bool(stage_cfg["sensor_dr"]),
                    "scenario_seed": int(scenario_seed),
                    "dr_seed": int(dr_seed),
                })
                active.append(worker_id)

            messages = []
            for worker_id in active:
                conn = parent_conns[worker_id]
                if not conn.poll(float(args.worker_timeout)):
                    raise TimeoutError("Worker {} rollout timeout.".format(worker_id))
                msg = conn.recv()
                if msg.get("type") == "error":
                    raise RuntimeError(msg.get("traceback", msg.get("error")))
                if msg.get("type") != "rollout":
                    raise RuntimeError("Unexpected worker message: {}".format(msg))
                messages.append(msg)

            collect_wall = time.time() - collect_start
            total_samples = sum(int(m["samples"]) for m in messages)
            if total_samples != requested_total:
                raise RuntimeError("Expected {} samples, got {}".format(requested_total, total_samples))

            diagnostics = aggregate_diagnostics(messages)
            episode_metrics = aggregate_episode_metrics(messages)

            if args.smoke_test:
                print("\nSMOKE PASS | maps={}".format([m["map_path"] for m in messages]))
                print_state_diagnostics(diagnostics)
                print("NO PPO UPDATE | NO CHECKPOINT")
                break

            master.memory.clear()
            merged = 0
            for msg in sorted(messages, key=lambda x: int(x["worker"])):
                merged += append_rollout_to_master(master, msg["rollout"])
            if merged != requested_total:
                raise RuntimeError("Merged {} != requested {}".format(merged, requested_total))

            ppo_metrics = master.learn(last_obs=None)
            global_step += requested_total
            episode += int(episode_metrics.get("episodes", 0))
            decay_bucket = maybe_decay_action_std(
                master, global_step, decay_bucket, args
            )

            print(
                "\nUPDATE {:05d} | step={}/{} | stage={} | block={} | maps={} | "
                "samples={} collect={:.1f}s SPS={:.2f} std={:.4f}".format(
                    update_index,
                    global_step,
                    target_step,
                    STAGE_NAMES[int(stage)],
                    stage_block_index,
                    [m["map_path"].split("/")[-1] for m in messages],
                    requested_total,
                    collect_wall,
                    float(requested_total) / max(collect_wall, 1e-9),
                    master.action_std,
                )
            )
            print(
                "  PPO | loss={:+.5f} policy={:+.5f} value={:.5f} entropy={:.5f} return={:+.5f}".format(
                    float(ppo_metrics.get("loss", 0.0)),
                    float(ppo_metrics.get("policy_loss", 0.0)),
                    float(ppo_metrics.get("value_loss", 0.0)),
                    float(ppo_metrics.get("entropy", 0.0)),
                    float(ppo_metrics.get("mean_return", 0.0)),
                )
            )
            print_state_diagnostics(diagnostics)

            if writer is not None:
                for key, value in ppo_metrics.items():
                    writer.add_scalar("ppo/{}".format(key), float(value), global_step)
                writer.add_scalar("curriculum/stage", float(stage), global_step)
                writer.add_scalar("curriculum/recovery_episode_prob", float(stage_cfg["recovery_prob"]), global_step)
                writer.add_scalar("perf/combined_sps", float(requested_total) / max(collect_wall, 1e-9), global_step)
                write_tensorboard_diagnostics(writer, diagnostics, global_step)

            # --------------------------------------------------------------
            # Guardrail canary + checkpoint gate.
            # MAIN numbered checkpoint creation happens ONLY inside PASS.
            # --------------------------------------------------------------
            if global_step >= next_canary_step or global_step >= target_step:
                evaluated_stage = int(stage)

                results = run_canary(parent_conns, master, args, global_step)
                passed, reasons, summary = judge_canary(
                    results, args, champion_payload=champion_payload
                )
                print_canary(results, passed, reasons, summary, global_step, stage)

                if writer is not None:
                    writer.add_scalar("canary/pass", 1.0 if passed else 0.0, global_step)
                    writer.add_scalar("canary/avg_abs_lateral_m", summary["avg_abs_lateral_m"], global_step)
                    writer.add_scalar("canary/max_abs_lateral_m", summary["max_abs_lateral_m"], global_step)
                    writer.add_scalar("canary/avg_abs_heading_deg", summary["avg_abs_heading_deg"], global_step)
                    writer.add_scalar("canary/edge_occupancy_rate", summary["edge_occupancy_rate"], global_step)
                    writer.add_scalar("canary/steer_positive_rate", summary["steer_positive_rate"], global_step)
                    writer.add_scalar("canary/steer_negative_rate", summary["steer_negative_rate"], global_step)

                if passed:
                    consecutive_passes += 1

                    # Decide curriculum state FIRST so resume state represents
                    # exactly what should run after this verified checkpoint.
                    stage_steps = int(global_step - stage_start_step)
                    if (
                        stage < 3
                        and stage_steps >= stage_min_steps(stage, args)
                        and consecutive_passes >= int(args.stage_pass_count)
                    ):
                        old_stage = int(stage)
                        stage += 1
                        stage_start_step = int(global_step)
                        consecutive_passes = 0
                        print(
                            "CURRICULUM ADVANCE | {} -> {} | step={}".format(
                                STAGE_NAMES[old_stage], STAGE_NAMES[stage], global_step
                            )
                        )

                    # GATE: this is the only normal numbered checkpoint save.
                    passed_path = save_passed_checkpoint(
                        master,
                        global_step,
                        episode,
                        stage,
                        stage_start_step,
                        consecutive_passes,
                        summary,
                        args,
                    )
                    last_valid_checkpoint_path = str(passed_path)

                    # Champion stores the exact rollback weights/optimizer from
                    # the policy that passed the canary. Record the stage it was
                    # evaluated under, even if curriculum just advanced.
                    save_champion(
                        master,
                        global_step,
                        evaluated_stage,
                        summary,
                        checkpoint_path_value=passed_path,
                    )
                    champion_payload = torch.load(
                        champion_path(), map_location=master.device
                    )

                    state = write_state(
                        agent=master,
                        global_step=global_step,
                        episode=episode,
                        stage=stage,
                        stage_start_step=stage_start_step,
                        consecutive_passes=consecutive_passes,
                        checkpoint_path_value=passed_path,
                        args=args,
                    )
                    print(
                        "CHAMPION UPDATE | {} | verified_policy={} | state={}".format(
                            champion_path(), passed_path, state
                        )
                    )

                else:
                    rejected = save_rejected_snapshot(
                        master,
                        global_step,
                        stage,
                        reasons,
                        summary,
                        keep=int(args.keep_rejected),
                    )
                    consecutive_passes = 0

                    if os.path.isfile(champion_path()):
                        print(
                            "CANDIDATE REJECTED | diagnostic only={} | NO main checkpoint created".format(
                                rejected
                            )
                        )

                        restored = load_champion(master)
                        champion_payload = restored

                        restored_cp = str(restored.get("checkpoint_path", ""))
                        if not restored_cp or not os.path.isfile(restored_cp):
                            raise RuntimeError(
                                "Champion rollback has no valid numbered checkpoint: {}".format(
                                    restored_cp
                                )
                            )
                        last_valid_checkpoint_path = restored_cp

                        # Restore action-std schedule from champion step up to the
                        # CURRENT global step. Otherwise rollback could silently
                        # resurrect an exploration std from an older step.
                        champion_bucket = int(
                            int(restored.get("global_step", 0))
                            // int(args.action_std_decay_freq)
                        )
                        decay_bucket = maybe_decay_action_std(
                            master,
                            global_step,
                            champion_bucket,
                            args,
                        )

                        print(
                            "ROLLBACK CHAMPION | champion_step={} champion_stage={} | policy={}".format(
                                restored.get("global_step"),
                                STAGE_NAMES[int(restored.get("stage", 0))],
                                restored_cp,
                            )
                        )

                        if stage > 0:
                            old_stage = int(stage)
                            stage -= 1
                            stage_start_step = int(global_step)
                            print(
                                "CURRICULUM FALLBACK | {} -> {}".format(
                                    STAGE_NAMES[old_stage], STAGE_NAMES[stage]
                                )
                            )
                        else:
                            stage = 0
                            stage_start_step = int(global_step)

                        # State-only persistence after rollback. It references
                        # the last VERIFIED numbered checkpoint; no new main
                        # checkpoint file is created on FAIL.
                        state = write_state(
                            agent=master,
                            global_step=global_step,
                            episode=episode,
                            stage=stage,
                            stage_start_step=stage_start_step,
                            consecutive_passes=consecutive_passes,
                            checkpoint_path_value=last_valid_checkpoint_path,
                            args=args,
                        )
                        print(
                            "ROLLBACK STATE | step={} | verified_policy={} | state={}".format(
                                global_step, last_valid_checkpoint_path, state
                            )
                        )
                    else:
                        # Fresh/random training cannot rollback before the first
                        # verified policy exists. Continue learning, but keep it
                        # explicitly UNVERIFIED and do not create a main policy.
                        stage = 0
                        stage_start_step = int(global_step)
                        print(
                            "CANARY FAIL | no champion exists yet | diagnostic={} | "
                            "continue Stage 0 UNVERIFIED | NO main checkpoint".format(
                                rejected
                            )
                        )

                while next_canary_step <= global_step:
                    next_canary_step += int(args.canary_every_steps)

        if not args.smoke_test:
            if not os.path.isfile(champion_path()):
                emergency = save_emergency_snapshot(
                    master,
                    global_step,
                    episode,
                    stage,
                    stage_start_step,
                    consecutive_passes,
                    args,
                    reason="training_complete_without_verified_champion",
                )
                print(
                    "WARNING | training reached target with NO verified champion. "
                    "Final candidate saved only as UNVERIFIED emergency: {}".format(
                        emergency
                    )
                )
            print(
                "\nTRAINING COMPLETE | step={} | episode={} | stage={} | elapsed={:.1f} min".format(
                    global_step, episode, STAGE_NAMES[int(stage)],
                    (time.time() - run_start) / 60.0,
                )
            )

    except KeyboardInterrupt:
        print("\nStopped by user.")
        if not args.smoke_test:
            try:
                emergency = save_emergency_snapshot(
                    master,
                    global_step,
                    episode,
                    stage,
                    stage_start_step,
                    consecutive_passes,
                    args,
                    reason="keyboard_interrupt",
                )
                print(
                    "EMERGENCY SNAPSHOT | UNVERIFIED | {} | "
                    "main/champion checkpoints unchanged".format(emergency)
                )
            except Exception:
                logging.exception("Could not save emergency snapshot.")

    except Exception:
        logging.exception("PPO V5 multi-map guarded trainer failed.")
        raise

    finally:
        for conn in parent_conns:
            try:
                conn.send({"cmd": "close"})
            except Exception:
                pass
        for process in processes:
            try:
                process.join(timeout=10.0)
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
        print("MULTIMAP GUARDED cleanup complete.")


if __name__ == "__main__":
    main()
