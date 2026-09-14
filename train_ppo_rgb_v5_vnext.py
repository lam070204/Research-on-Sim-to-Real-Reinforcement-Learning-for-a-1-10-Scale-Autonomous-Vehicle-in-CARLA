# -*- coding: utf-8 -*-
"""
PPO V5 VNEXT MAIN TRAINER
=============================================================

Why this trainer exists
-----------------------
The previous guarded trainer protected the two anchor maps very well, but its
checkpoint gate only evaluated the anchors.  A policy could therefore be
"clean-canary verified" while one or more auxiliary maps still contained a
systematic dead spot.  This trainer keeps the tested V5 PPO/environment path,
but adds the missing supervision layers:

1) clean anchor canary (fast, frequent),
2) per-map clean mastery validation (all maps, all configured spawns),
3) adaptive map sampling (FAIL/WEAK maps get more training, PASS maps never 0),
4) recovery exposure/success/speed-state audit,
5) cumulative Dynamics/Vision/Sensor/Full-DR exposure audit,
6) DR parameter coverage audit (range bins + per-map DR episode counts),
7) robust DR validation across every map,
8) lightweight failure-position clustering,
9) separate CLEAN / MULTIMAP / RECOVERY / SIM2REAL champions,
10) no automatic stage decrement that silently starves Full-DR exposure,
11) 100k-transition interleaved map blocks with adaptive mastery-based percentages.

The V5 policy architecture, obs100, reward path, clean spawn rule, delayed
recovery and environment action semantics are NOT changed here.

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
from collections import defaultdict
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
    SEED,
)
from simulation.carla_connection_v5 import carla
from simulation.carla_environment_rgb_v5_multimap import (
    CarlaEnvironmentRGBV5MultiMap,
    configure_recovery_scenarios_v5,
)
from vehicle_specs_v5 import CARLA_GEOMETRY_SCALE
from domain_randomization_v5 import (
    DOMAIN_RANDOMIZATION_RANGES,
    CAMERA_REAL_OFFSET_RANGES_M,
    CAMERA_ANGLE_OFFSET_RANGES_DEG,
    CAMERA_FOV_OFFSET_DEG,
    WEATHER_RANGES,
    IMAGE_EPISODE_RANGES,
    STATE_SENSOR_RANGES,
)

# Reuse pure rollout/PPO diagnostic helpers already used by the production
# guarded trainer.  Training control logic is implemented in this file.
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


# =============================================================================
# Maps / stages
# =============================================================================

ANCHOR_MAPS = (
    "/Game/mapxanh/xanhphai",
    "/Game/xanhtrai/mapxanhdam",
)

AUX_OVAL_MAPS = (
    "/Game/dancu_phai/dancu_phai",
    "/Game/dancu_trai/dancu_trai",
    "/Game/maplonxon/maplonxon",
    "/Game/maplonxon1/maplonxon1",
    "/Game/rungphai/rung_phai",
    "/Game/rungtrai/rung_trai",
    "/Game/maptotrai/mapto_trai",
    "/Game/maptophai/mapto_phai",
)

LEFT_ONLY_STRESS_MAP = "/Game/maptrangcorao/maptuong"
ALL_MAPS = tuple(dict.fromkeys(ANCHOR_MAPS + AUX_OVAL_MAPS + (LEFT_ONLY_STRESS_MAP,)))

# Hard safety guards: VNext must always use the two mapto maps as clean anchors,
# and the full pool must contain exactly 11 unique maps.  This prevents a bad
# local edit from silently turning AUX maps into the ANCHOR canary or duplicating
# maps in mastery/scheduler tables.
_EXPECTED_ANCHORS = (
    "/Game/mapxanh/xanhphai",
    "/Game/xanhtrai/mapxanhdam",
)
if tuple(ANCHOR_MAPS) != _EXPECTED_ANCHORS:
    raise RuntimeError("VNEXT MAP CONFIG ERROR: ANCHOR_MAPS must be exactly {} but got {}".format(
        _EXPECTED_ANCHORS, ANCHOR_MAPS
    ))
if len(ALL_MAPS) != 11 or len(set(ALL_MAPS)) != 11:
    raise RuntimeError("VNEXT MAP CONFIG ERROR: expected 11 unique maps, got {} entries / {} unique: {}".format(
        len(ALL_MAPS), len(set(ALL_MAPS)), ALL_MAPS
    ))


def _norm_map_token(value):
    return "".join(ch for ch in str(value).replace("\\\\", "/").lower() if ch.isalnum())


def resolve_map_path_from_available(requested, available_maps):
    """Resolve a configured CARLA map path against client.get_available_maps().

    Exact path wins. If the folder/package path changed between CARLA builds, a
    unique basename/fuzzy suffix match is accepted. Ambiguous or absent maps are
    never silently skipped.
    """
    requested = str(requested).replace("\\\\", "/")
    available = [str(x).replace("\\\\", "/") for x in (available_maps or [])]
    req_low = requested.lower().rstrip("/")
    exact = [x for x in available if x.lower().rstrip("/") == req_low]
    if len(exact) == 1:
        return exact[0], []

    req_tail = requested.rstrip("/").split("/")[-1]
    req_tail_norm = _norm_map_token(req_tail)
    scored = []
    for item in available:
        tail = item.rstrip("/").split("/")[-1]
        tail_norm = _norm_map_token(tail)
        path_norm = _norm_map_token(item)
        score = 0
        if tail.lower() == req_tail.lower():
            score = 100
        elif tail_norm == req_tail_norm and req_tail_norm:
            score = 95
        elif req_tail_norm and (tail_norm.endswith(req_tail_norm) or req_tail_norm.endswith(tail_norm)):
            score = 80
        elif req_tail_norm and req_tail_norm in path_norm:
            score = 70
        if score:
            scored.append((score, item))

    if not scored:
        # Helpful diagnostics only; these are not accepted as a resolution.
        hints = []
        req_words = [w for w in ("xanh", "trai", "phai", "rung", "dancu", "lonxon", "tuong") if w in req_low]
        for item in available:
            low = item.lower()
            if any(w in low for w in req_words):
                hints.append(item)
        return None, hints[:20]

    best = max(score for score, _ in scored)
    best_items = sorted({item for score, item in scored if score == best})
    if len(best_items) == 1 and best >= 80:
        return best_items[0], []
    return None, best_items[:20]

MAP_GROUP = {}
for _m in ANCHOR_MAPS:
    MAP_GROUP[_m] = "anchor"
for _m in AUX_OVAL_MAPS:
    MAP_GROUP[_m] = "aux"
MAP_GROUP[LEFT_ONLY_STRESS_MAP] = "stress"

STAGE_NAMES = {
    0: "ANCHOR_NOMINAL",
    1: "ANCHOR_DELAYED_RECOVERY",
    2: "MULTIMAP_MASTERY",
    3: "MULTIMAP_FULL_SIM2REAL",
}

MODEL_NAME = "automav5_rgb_vnext"
TRAINER_VERSION = "TRAINER_PPO_V5_VNEXT_MAIN_6_ANCHOR_GUARD_ROLLBACK_FIX"
LEGACY_STATE_VERSIONS = (
    "TRAINER_PPO_V5_VNEXT_MAIN_3",
    "TRAINER_PPO_V5_VNEXT_MAIN_4_100K_ADAPTIVE",
    "TRAINER_PPO_V5_VNEXT_MAIN_5_100K_ADAPTIVE_MAP_RESOLVE",
)
STATE_NAME = "training_state_vnext.json"
CONTROL_HZ = 50.0

CHAMPION_FILES = {
    "clean": "champion_clean.pth",
    "multimap": "champion_multimap.pth",
    "recovery": "champion_recovery.pth",
    "sim2real": "champion_sim2real.pth",
}

STATUS_UNKNOWN = "UNKNOWN"
STATUS_FAIL = "FAIL"
STATUS_WEAK = "WEAK"
STATUS_PASS = "PASS"


def map_short(map_path):
    return str(map_path).replace("\\", "/").split("/")[-1]


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
        value = int(item)
        if value <= 0:
            raise argparse.ArgumentTypeError("Spawn numbers must be > 0.")
        values.append(value)
    values = list(dict.fromkeys(values))
    if not values:
        raise argparse.ArgumentTypeError("At least one spawn is required.")
    return values


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PPO V5 VNext: per-map mastery + adaptive scheduler + DR coverage gates."
    )

    # CARLA workers.
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port0", type=int, default=2000)
    p.add_argument("--port1", type=int, default=2010)
    p.add_argument("--carla-timeout", type=float, default=180.0)
    p.add_argument("--worker-timeout", type=float, default=1800.0)

    # Start/resume.  init-stage>0 is allowed only with an existing checkpoint.
    p.add_argument("--resume", type=boolean_string, default=False)
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--init-stage", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--resume-policy-override", default="",
                   help="On --resume, keep training state/counters but load policy weights from this checkpoint (rescue use only).")
    p.add_argument("--force-stage3-now", type=boolean_string, default=False,
                   help="On --resume from Stage2, immediately persist Stage3 and enable FULL Sim2Real DR without waiting for the clean mastery gate.")
    p.add_argument("--total-timesteps", type=int, default=2000000)
    p.add_argument("--rollout-total", type=int, default=1024)
    p.add_argument("--map-block-steps", type=int, default=100000)
    p.add_argument("--safe-spawns", type=parse_spawn_numbers,
                   default=parse_spawn_numbers("1,2,3,4"))
    p.add_argument("--desired-speed", type=float, default=1.0)
    p.add_argument("--max-episode-seconds", type=float, default=60.0)

    # Recovery - clean spawn, delayed injection only.
    p.add_argument("--recovery-episode-prob", type=float, default=0.30)
    p.add_argument("--recovery-min-episode-step", type=int, default=700)
    p.add_argument("--recovery-stable-ticks", type=int, default=100)
    p.add_argument("--recovery-stable-lateral-m", type=float, default=0.040)
    p.add_argument("--recovery-stable-heading-deg", type=float, default=5.0)
    p.add_argument("--recovery-stable-speed-mps", type=float, default=0.40)
    p.add_argument("--recovery-cooldown-ticks", type=int, default=400)
    p.add_argument("--recovery-max-events", type=int, default=1)

    # Progressive recovery levels. Recovery steer amplitude intentionally reduced to 50%; durations/probability unchanged.
    p.add_argument("--recovery-level1-steer-min", type=float, default=0.05)
    p.add_argument("--recovery-level1-steer-max", type=float, default=0.09)
    p.add_argument("--recovery-level1-duration-min-s", type=float, default=0.10)
    p.add_argument("--recovery-level1-duration-max-s", type=float, default=0.16)
    p.add_argument("--recovery-level2-steer-min", type=float, default=0.08)
    p.add_argument("--recovery-level2-steer-max", type=float, default=0.12)
    p.add_argument("--recovery-level2-duration-min-s", type=float, default=0.12)
    p.add_argument("--recovery-level2-duration-max-s", type=float, default=0.18)
    p.add_argument("--recovery-level3-steer-min", type=float, default=0.10)
    p.add_argument("--recovery-level3-steer-max", type=float, default=0.15)
    p.add_argument("--recovery-level3-duration-min-s", type=float, default=0.14)
    p.add_argument("--recovery-level3-duration-max-s", type=float, default=0.20)
    p.add_argument("--recovery-level-max", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--recovery-level-min-events-per-sign", type=int, default=8)
    p.add_argument("--recovery-level-success-rate", type=float, default=0.85)
    p.add_argument("--recovery-level-side-success-rate", type=float, default=0.75)

    # DR switches are honored only in Stage 3 (outside temporary repair mode).
    p.add_argument("--dynamics-dr", type=boolean_string, default=True)
    p.add_argument("--vision-dr", type=boolean_string, default=True)
    p.add_argument("--sensor-dr", type=boolean_string, default=True)

    # Stage minimums.  Stage2->3 additionally requires per-map mastery.
    p.add_argument("--stage0-min-steps", type=int, default=50000)
    p.add_argument("--stage1-min-steps", type=int, default=100000)
    p.add_argument("--stage2-min-steps", type=int, default=200000)
    p.add_argument("--stage-pass-count", type=int, default=2)

    # Frequent strict anchor canary.
    p.add_argument("--canary-every-steps", type=int, default=20000)
    p.add_argument("--canary-max-steps", type=int, default=750)
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

    # Full clean per-map mastery gate.
    p.add_argument("--initial-map-validation", type=boolean_string, default=True)
    p.add_argument("--map-validation-every-steps", type=int, default=100000)
    p.add_argument("--mastery-max-steps", type=int, default=1000)
    p.add_argument("--mastery-spawns", type=parse_spawn_numbers,
                   default=parse_spawn_numbers("1,2,3,4"))
    p.add_argument("--mastery-pass-count", type=int, default=2)
    p.add_argument("--mastery-avg-lateral-max-m", type=float, default=0.035)
    p.add_argument("--mastery-max-lateral-max-m", type=float, default=0.070)
    p.add_argument("--mastery-avg-heading-max-deg", type=float, default=7.0)
    p.add_argument("--mastery-edge-rate-max", type=float, default=0.08)
    p.add_argument("--mastery-steer075-rate-max", type=float, default=0.10)
    p.add_argument("--mastery-min-progress-m", type=float, default=3.0)
    p.add_argument("--stress-required-for-stage3", type=boolean_string, default=True)

    # GLOBAL adaptive map scheduler.  No fixed 50/45/5 group mass.
    # Hard/unfinished maps receive more probability; mastered maps keep a floor
    # so the shared PPO does not catastrophically forget them.
    p.add_argument("--map-priority-fail", type=float, default=4.00)
    p.add_argument("--map-priority-weak", type=float, default=2.50)
    p.add_argument("--map-priority-unknown", type=float, default=2.00)
    p.add_argument("--map-priority-pass", type=float, default=0.60)
    p.add_argument("--map-priority-mastered", type=float, default=0.20)
    p.add_argument("--map-floor-anchor", type=float, default=0.05)
    p.add_argument("--map-floor-other", type=float, default=0.02)
    p.add_argument("--map-max-weight", type=float, default=0.25)
    p.add_argument("--stress-max-weight", type=float, default=0.08)
    p.add_argument("--failure-cluster-weight-boost", type=float, default=1.50)
    p.add_argument("--robust-fail-weight-boost", type=float, default=1.50)
    p.add_argument("--robust-weak-weight-boost", type=float, default=1.20)
    p.add_argument("--scheduler-cycle-slots", type=int, default=100)

    # Recovery gate from actual training injections.
    p.add_argument("--recovery-gate-min-events-per-sign", type=int, default=8)
    p.add_argument("--recovery-gate-success-rate", type=float, default=0.80)
    p.add_argument("--recovery-gate-side-success-rate", type=float, default=0.70)

    # Robust DR validation. Every map is sampled; repeated resets produce
    # deterministic but different DR episode samples from the given seed.
    p.add_argument("--robust-validation-every-steps", type=int, default=100000)
    p.add_argument("--robust-max-steps", type=int, default=750)
    p.add_argument("--robust-spawns", type=parse_spawn_numbers,
                   default=parse_spawn_numbers("1,3"))
    p.add_argument("--robust-repeats", type=int, default=2)
    p.add_argument("--robust-map-pass-rate", type=float, default=0.75)
    p.add_argument("--robust-avg-lateral-max-m", type=float, default=0.045)
    p.add_argument("--robust-max-lateral-max-m", type=float, default=0.090)
    p.add_argument("--robust-avg-heading-max-deg", type=float, default=9.0)
    p.add_argument("--robust-steer075-rate-max", type=float, default=0.15)
    p.add_argument("--robust-edge-rate-max", type=float, default=0.12)
    p.add_argument("--robust-min-progress-m", type=float, default=2.0)

    # DR exposure/coverage gate. Ratio is relative to transitions produced by
    # this VNext run (global_step starts at 0 for --init-checkpoint fine-tune).
    p.add_argument("--dr-min-full-steps", type=int, default=250000)
    p.add_argument("--dr-min-full-ratio", type=float, default=0.20)
    p.add_argument("--dr-min-episode-samples", type=int, default=100)
    p.add_argument("--dr-min-episodes-per-map", type=int, default=4)
    p.add_argument("--dr-coverage-min-fraction", type=float, default=0.60)
    p.add_argument("--dr-coverage-bins", type=int, default=8)

    # Stage3 clean-regression repair: stay in Stage3, temporarily train clean
    # multimap instead of dropping an entire stage and resetting a 200k clock.
    p.add_argument("--stage3-anchor-fail-patience", type=int, default=3)
    p.add_argument("--stage3-repair-steps", type=int, default=50000)

    # Failure clustering (CARLA world coordinates; geometry is 10x real).
    p.add_argument("--failure-cluster-grid-carla-m", type=float, default=1.5)
    p.add_argument("--failure-cluster-alert-count", type=int, default=5)

    # Latent95 supervision. This is diagnostic only by default: it never
    # perturbs the observation used for training. The sensitivity probe asks
    # how much deterministic steer changes for a very small latent-only move.
    p.add_argument("--latent-audit", type=boolean_string, default=True)
    p.add_argument("--latent-sensitivity-every-steps", type=int, default=50)
    p.add_argument("--latent-sensitivity-epsilon-l2", type=float, default=0.05)
    p.add_argument("--latent-sensitivity-alert", type=float, default=2.0)

    # Checkpoint housekeeping.
    p.add_argument("--keep-rejected", type=int, default=8)
    p.add_argument("--keep-emergency", type=int, default=3)

    # PPO / devices.
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
    p.add_argument("--action-std-decay-freq", type=int,
                   default=int(PPO_ACTION_STD_DECAY_FREQ))
    p.add_argument("--tensorboard", type=boolean_string, default=True)
    p.add_argument("--smoke-test", type=boolean_string, default=False)
    p.add_argument("--smoke-steps-per-worker", type=int, default=64)
    return p.parse_args()


def validate_args(a):
    if int(a.workers) == 2 and int(a.port0) == int(a.port1):
        raise ValueError("port0 and port1 must differ")
    if bool(a.resume) and str(a.init_checkpoint).strip():
        raise ValueError("Use either --resume true OR --init-checkpoint")
    if int(a.init_stage) > 0 and not (bool(a.resume) or str(a.init_checkpoint).strip()):
        raise ValueError("--init-stage > 0 requires --init-checkpoint or --resume")
    if int(a.total_timesteps) <= 0 or int(a.rollout_total) < int(a.workers):
        raise ValueError("Invalid total/rollout steps")
    if abs(float(a.desired_speed) - 1.0) > 1e-9:
        raise ValueError("V5 state-speed profile is locked to cruise=1.0 m/s")
    if not (0.0 <= float(a.recovery_episode_prob) <= 1.0):
        raise ValueError("recovery episode probability must be in [0,1]")
    if not (0.0 <= float(a.map_floor_anchor) < 1.0):
        raise ValueError("--map-floor-anchor must be in [0,1)")
    if not (0.0 <= float(a.map_floor_other) < 1.0):
        raise ValueError("--map-floor-other must be in [0,1)")
    if not (0.0 < float(a.map_max_weight) <= 1.0):
        raise ValueError("--map-max-weight must be in (0,1]")
    if not (0.0 < float(a.stress_max_weight) <= 1.0):
        raise ValueError("--stress-max-weight must be in (0,1]")
    minimum_mass = (
        len(ANCHOR_MAPS) * float(a.map_floor_anchor)
        + (len(ALL_MAPS) - len(ANCHOR_MAPS)) * float(a.map_floor_other)
    )
    if minimum_mass >= 1.0:
        raise ValueError("Map probability floors consume >=100% mass")
    for name in (
        "map_priority_fail", "map_priority_weak", "map_priority_unknown",
        "map_priority_pass", "map_priority_mastered",
        "failure_cluster_weight_boost", "robust_fail_weight_boost",
        "robust_weak_weight_boost",
    ):
        if float(getattr(a, name)) <= 0.0:
            raise ValueError("--{} must be > 0".format(name.replace("_", "-")))
    for name in (
        "map_validation_every_steps", "robust_validation_every_steps",
        "canary_every_steps", "mastery_max_steps", "robust_max_steps",
        "scheduler_cycle_slots",
    ):
        if int(getattr(a, name)) <= 0:
            raise ValueError("--{} must be > 0".format(name.replace("_", "-")))
    if int(a.mastery_pass_count) <= 0 or int(a.stage_pass_count) <= 0:
        raise ValueError("pass counts must be > 0")
    if int(a.robust_repeats) <= 0:
        raise ValueError("--robust-repeats must be > 0")
    if not (0.0 <= float(a.dr_min_full_ratio) <= 1.0):
        raise ValueError("--dr-min-full-ratio must be in [0,1]")
    if not (0.0 <= float(a.dr_coverage_min_fraction) <= 1.0):
        raise ValueError("--dr-coverage-min-fraction must be in [0,1]")
    if int(a.latent_sensitivity_every_steps) <= 0:
        raise ValueError("--latent-sensitivity-every-steps must be > 0")
    if float(a.latent_sensitivity_epsilon_l2) <= 0.0:
        raise ValueError("--latent-sensitivity-epsilon-l2 must be > 0")
    if float(a.latent_sensitivity_alert) <= 0.0:
        raise ValueError("--latent-sensitivity-alert must be > 0")


# =============================================================================
# Stage/recovery configuration
# =============================================================================

def stage_min_steps(stage, args):
    return {
        0: int(args.stage0_min_steps),
        1: int(args.stage1_min_steps),
        2: int(args.stage2_min_steps),
        3: 10 ** 18,
    }[int(stage)]


def stage_config(stage, args, repair_mode=False, recovery_level=1):
    stage = int(stage)
    if stage == 0:
        return dict(name=STAGE_NAMES[stage], multimap=False, recovery=False,
                    recovery_prob=0.0, dynamics_dr=False, vision_dr=False,
                    sensor_dr=False, repair=False, recovery_level=1)
    if stage == 1:
        return dict(name=STAGE_NAMES[stage], multimap=False, recovery=True,
                    recovery_prob=float(args.recovery_episode_prob),
                    dynamics_dr=False, vision_dr=False, sensor_dr=False,
                    repair=False, recovery_level=int(recovery_level))
    if stage == 2:
        return dict(name=STAGE_NAMES[stage], multimap=True, recovery=True,
                    recovery_prob=float(args.recovery_episode_prob),
                    dynamics_dr=False, vision_dr=False, sensor_dr=False,
                    repair=False, recovery_level=int(recovery_level))
    if repair_mode:
        return dict(name=STAGE_NAMES[stage] + "_REPAIR", multimap=True,
                    recovery=True, recovery_prob=float(args.recovery_episode_prob),
                    dynamics_dr=False, vision_dr=False, sensor_dr=False,
                    repair=True, recovery_level=int(recovery_level))
    return dict(name=STAGE_NAMES[stage], multimap=True, recovery=True,
                recovery_prob=float(args.recovery_episode_prob),
                dynamics_dr=bool(args.dynamics_dr),
                vision_dr=bool(args.vision_dr), sensor_dr=bool(args.sensor_dr),
                repair=False, recovery_level=int(recovery_level))


def recovery_level_params(level, args):
    level = max(1, min(int(args.recovery_level_max), int(level)))
    prefix = "recovery_level{}".format(level)
    return {
        "level": level,
        "steer_min": float(getattr(args, prefix + "_steer_min")),
        "steer_max": float(getattr(args, prefix + "_steer_max")),
        "duration_min_s": float(getattr(args, prefix + "_duration_min_s")),
        "duration_max_s": float(getattr(args, prefix + "_duration_max_s")),
    }


# =============================================================================
# Per-map mastery / adaptive scheduling
# =============================================================================

class MapMasteryTracker(object):
    def __init__(self):
        self.maps = {}
        for map_path in ALL_MAPS:
            self.maps[map_path] = {
                "status": STATUS_UNKNOWN,
                "pass_streak": 0,
                "last_eval_step": -1,
                "pass_rate": 0.0,
                "cases": 0,
                "failed_cases": 0,
                "last_summary": {},
                "robust_status": STATUS_UNKNOWN,
                "robust_pass_rate": 0.0,
                "robust_cases": 0,
                "robust_last_eval_step": -1,
            }

    def to_dict(self):
        return copy.deepcopy(self.maps)

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        if isinstance(data, dict):
            for map_path, values in data.items():
                if map_path in obj.maps and isinstance(values, dict):
                    obj.maps[map_path].update(values)
        return obj

    def status(self, map_path):
        return str(self.maps.get(map_path, {}).get("status", STATUS_UNKNOWN))

    def pass_streak(self, map_path):
        return int(self.maps.get(map_path, {}).get("pass_streak", 0))

    def update(self, results, args, global_step):
        grouped = defaultdict(list)
        for item in results:
            grouped[str(item["map_path"])].append(item)

        reports = {}
        for map_path in ALL_MAPS:
            items = grouped.get(map_path, [])
            case_flags = [judge_eval_case(x, args, mode="mastery")[0] for x in items]
            cases = len(items)
            passed_cases = int(sum(1 for x in case_flags if x))
            pass_rate = float(passed_cases) / float(cases) if cases else 0.0

            if cases > 0 and passed_cases == cases:
                status = STATUS_PASS
            elif cases > 0 and pass_rate >= 0.75:
                status = STATUS_WEAK
            else:
                status = STATUS_FAIL

            record = self.maps[map_path]
            if status == STATUS_PASS:
                record["pass_streak"] = int(record.get("pass_streak", 0)) + 1
            else:
                record["pass_streak"] = 0
            record["status"] = status
            record["last_eval_step"] = int(global_step)
            record["pass_rate"] = float(pass_rate)
            record["cases"] = int(cases)
            record["failed_cases"] = int(cases - passed_cases)
            summary = summarize_eval_cases(items)
            record["last_summary"] = summary
            reports[map_path] = copy.deepcopy(record)
        return reports

    def update_robust(self, map_reports, args, global_step):
        if not isinstance(map_reports, dict):
            return
        threshold = float(args.robust_map_pass_rate)
        weak_threshold = max(0.50, threshold - 0.20)
        for map_path in ALL_MAPS:
            report = map_reports.get(map_path, {})
            rate = float(report.get("pass_rate", 0.0)) if isinstance(report, dict) else 0.0
            cases = int(report.get("cases", 0)) if isinstance(report, dict) else 0
            if cases <= 0:
                status = STATUS_UNKNOWN
            elif rate >= threshold:
                status = STATUS_PASS
            elif rate >= weak_threshold:
                status = STATUS_WEAK
            else:
                status = STATUS_FAIL
            rec = self.maps[map_path]
            rec["robust_status"] = status
            rec["robust_pass_rate"] = rate
            rec["robust_cases"] = cases
            rec["robust_last_eval_step"] = int(global_step)

    def required_maps_mastered(self, args):
        required = list(ANCHOR_MAPS) + list(AUX_OVAL_MAPS)
        if bool(args.stress_required_for_stage3):
            required.append(LEFT_ONLY_STRESS_MAP)
        missing = []
        for map_path in required:
            rec = self.maps[map_path]
            if (
                rec.get("status") != STATUS_PASS
                or int(rec.get("pass_streak", 0)) < int(args.mastery_pass_count)
            ):
                missing.append(map_path)
        return (len(missing) == 0), missing


def map_is_mastered(mastery, map_path, args):
    rec = mastery.maps.get(map_path, {})
    return bool(
        rec.get("status") == STATUS_PASS
        and int(rec.get("pass_streak", 0)) >= int(args.mastery_pass_count)
    )


def map_priority(mastery, map_path, args, cluster_maps):
    rec = mastery.maps.get(map_path, {})
    status = str(rec.get("status", STATUS_UNKNOWN))

    if map_is_mastered(mastery, map_path, args):
        priority = float(args.map_priority_mastered)
    elif status == STATUS_FAIL:
        priority = float(args.map_priority_fail)
    elif status == STATUS_WEAK:
        priority = float(args.map_priority_weak)
    elif status == STATUS_PASS:
        priority = float(args.map_priority_pass)
    else:
        priority = float(args.map_priority_unknown)

    # Continuous difficulty boost: a 0%-pass map receives more pressure than
    # a map that only narrowly misses the gate.
    pass_rate = max(0.0, min(1.0, float(rec.get("pass_rate", 0.0))))
    if status in (STATUS_FAIL, STATUS_WEAK):
        priority *= (1.0 + 0.75 * (1.0 - pass_rate))

    # Repeated offroad at the same world region is evidence of a systematic
    # dead spot, not random exploration noise.
    if map_path in cluster_maps:
        priority *= float(args.failure_cluster_weight_boost)

    robust_status = str(rec.get("robust_status", STATUS_UNKNOWN))
    if robust_status == STATUS_FAIL:
        priority *= float(args.robust_fail_weight_boost)
    elif robust_status == STATUS_WEAK:
        priority *= float(args.robust_weak_weight_boost)

    return max(float(priority), 1e-6)


def _allocate_probability_with_floors_caps(priorities, floors, caps):
    """Allocate exactly 1.0 probability mass with per-map floors/caps."""
    weights = {m: float(floors[m]) for m in ALL_MAPS}
    remaining = 1.0 - float(sum(weights.values()))
    if remaining < -1e-9:
        raise RuntimeError("Map probability floors exceed 1.0")

    active = set(ALL_MAPS)
    for _ in range(len(ALL_MAPS) + 2):
        if remaining <= 1e-12 or not active:
            break
        total_priority = sum(float(priorities[m]) for m in active)
        if total_priority <= 0.0:
            total_priority = float(len(active))
            shares = {m: remaining / float(len(active)) for m in active}
        else:
            shares = {
                m: remaining * float(priorities[m]) / total_priority
                for m in active
            }

        saturated = []
        for m in list(active):
            room = max(0.0, float(caps[m]) - float(weights[m]))
            if shares[m] >= room - 1e-12:
                weights[m] += room
                remaining -= room
                saturated.append(m)

        if not saturated:
            for m in active:
                weights[m] += shares[m]
            remaining = 0.0
            break

        for m in saturated:
            active.discard(m)

    # Numerical residue only. Put it into maps with remaining room.
    if remaining > 1e-9:
        for m in ALL_MAPS:
            room = max(0.0, float(caps[m]) - float(weights[m]))
            add = min(room, remaining)
            weights[m] += add
            remaining -= add
            if remaining <= 1e-9:
                break

    total = float(sum(weights.values()))
    if abs(total - 1.0) > 1e-6:
        raise RuntimeError(
            "Could not allocate map probabilities: sum={:.9f}, residual={:.9f}".format(
                total, remaining
            )
        )
    return weights


def adaptive_map_weights(mastery, args, cluster_maps=None):
    """
    GLOBAL adaptive scheduler.

    There is intentionally no fixed Anchor/Aux group mass. A difficult
    auxiliary map is allowed to receive more probability than an already
    mastered anchor. Mastered anchors retain a floor to prevent forgetting.
    """
    cluster_maps = set(cluster_maps or [])
    priorities = {
        m: map_priority(mastery, m, args, cluster_maps)
        for m in ALL_MAPS
    }
    floors = {
        m: (
            float(args.map_floor_anchor)
            if m in ANCHOR_MAPS
            else float(args.map_floor_other)
        )
        for m in ALL_MAPS
    }
    caps = {
        m: (
            float(args.stress_max_weight)
            if m == LEFT_ONLY_STRESS_MAP
            else float(args.map_max_weight)
        )
        for m in ALL_MAPS
    }
    return _allocate_probability_with_floors_caps(priorities, floors, caps)

def _slot_counts_from_weights(weights, nslots):
    nslots = int(nslots)
    raw = {m: float(weights[m]) * nslots for m in ALL_MAPS}
    counts = {m: int(math.floor(raw[m])) for m in ALL_MAPS}
    remaining = nslots - sum(counts.values())
    order = sorted(ALL_MAPS, key=lambda m: (raw[m] - counts[m]), reverse=True)
    for m in order[:remaining]:
        counts[m] += 1
    return counts


def weighted_cycle(weights, cycle_index, nslots, worker_count=1):
    """
    Build a deterministic INTERLEAVED weighted cycle.

    Unlike random.shuffle(), repeated slots for a high-priority map are spread
    across the cycle.  The per-map slot counts still come directly from the
    adaptive probabilities, so FAIL/WEAK maps appear more often while
    PASS/MASTERED maps automatically appear less often.
    """
    nslots = int(nslots)
    worker_count = max(1, int(worker_count))
    counts = _slot_counts_from_weights(weights, nslots)
    used = {m: 0 for m in ALL_MAPS}
    remaining = dict(counts)
    cycle = []

    # Rotate tie-breaking between cycles so equal-weight maps do not always
    # receive the same earliest slots.
    offset = int(cycle_index) % max(1, len(ALL_MAPS))
    tie_order = list(ALL_MAPS[offset:]) + list(ALL_MAPS[:offset])
    tie_rank = {m: i for i, m in enumerate(tie_order)}

    for slot in range(nslots):
        candidates = [m for m in ALL_MAPS if remaining[m] > 0]
        if not candidates:
            break

        # Prefer a different map for workers that run in parallel in the same
        # 100k block whenever enough candidates exist.
        block_pos = slot % worker_count
        current_block = set(cycle[-block_pos:]) if block_pos > 0 else set()
        diverse = [m for m in candidates if m not in current_block]
        if diverse:
            candidates = diverse

        def score(map_path):
            # Largest cumulative scheduling deficit wins.  This is a smooth
            # weighted-round-robin rule that spreads high-count maps instead
            # of placing them in long consecutive runs.
            target_used = float(slot + 1) * float(counts[map_path]) / float(nslots)
            deficit = target_used - float(used[map_path])
            return (deficit, -tie_rank[map_path])

        chosen = max(candidates, key=score)
        cycle.append(chosen)
        used[chosen] += 1
        remaining[chosen] -= 1

    if len(cycle) != nslots:
        raise RuntimeError("Weighted interleaved cycle length mismatch: {} != {}".format(len(cycle), nslots))
    return cycle


def assigned_maps_for_block(stage, block_index, worker_count, weights, args):
    """Return one fixed map per worker for the whole map block.

    Stage0/1 remain anchor-only. Stage2/3 use the adaptive weighted cycle.
    The caller caches this result, therefore a worker does not change maps
    inside a 100k block even if failure statistics change mid-block.
    """
    stage = int(stage)
    block_index = int(block_index)
    worker_count = max(1, int(worker_count))

    if stage <= 1:
        return [
            ANCHOR_MAPS[(block_index + worker_id) % len(ANCHOR_MAPS)]
            for worker_id in range(worker_count)
        ]

    cycle_len = int(args.scheduler_cycle_slots)
    first_global_slot = block_index * worker_count
    cycle_index = first_global_slot // cycle_len
    cycle = weighted_cycle(weights, cycle_index, cycle_len, worker_count=worker_count)

    result = []
    used_this_block = set()
    for worker_id in range(worker_count):
        global_slot = first_global_slot + worker_id
        position = global_slot % cycle_len
        candidate = cycle[position]

        # If the cycle boundary happens to give both workers the same map,
        # scan forward to a different eligible map for this block.
        if candidate in used_this_block and len(ALL_MAPS) > len(used_this_block):
            for delta in range(1, cycle_len):
                alt = cycle[(position + delta) % cycle_len]
                if alt not in used_this_block:
                    candidate = alt
                    break

        result.append(candidate)
        used_this_block.add(candidate)
    return result


def assigned_map(stage, block_index, worker_id, worker_count, weights, args):
    # Backward-compatible helper for any external caller.
    return assigned_maps_for_block(
        stage, block_index, worker_count, weights, args
    )[int(worker_id)]


# =============================================================================
# Recovery audit
# =============================================================================

class RecoveryAudit(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.injected_pos = 0
        self.injected_neg = 0
        self.success_pos = 0
        self.success_neg = 0
        self.state_counts = {s: 0 for s in ("stable", "mild", "moderate", "strong")}
        self.state_cmd_error_sum = {s: 0.0 for s in self.state_counts}
        self.state_speed_error_sum = {s: 0.0 for s in self.state_counts}

    def update_export(self, data):
        if not isinstance(data, dict):
            return
        for name in ("injected_pos", "injected_neg", "success_pos", "success_neg"):
            setattr(self, name, int(getattr(self, name)) + int(data.get(name, 0)))
        for state in self.state_counts:
            self.state_counts[state] += int(data.get("state_counts", {}).get(state, 0))
            self.state_cmd_error_sum[state] += float(data.get("state_cmd_error_sum", {}).get(state, 0.0))
            self.state_speed_error_sum[state] += float(data.get("state_speed_error_sum", {}).get(state, 0.0))

    def to_dict(self):
        return {
            "injected_pos": int(self.injected_pos),
            "injected_neg": int(self.injected_neg),
            "success_pos": int(self.success_pos),
            "success_neg": int(self.success_neg),
            "state_counts": dict(self.state_counts),
            "state_cmd_error_sum": dict(self.state_cmd_error_sum),
            "state_speed_error_sum": dict(self.state_speed_error_sum),
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        if not isinstance(data, dict):
            return obj
        obj.injected_pos = int(data.get("injected_pos", 0))
        obj.injected_neg = int(data.get("injected_neg", 0))
        obj.success_pos = int(data.get("success_pos", 0))
        obj.success_neg = int(data.get("success_neg", 0))
        for state in obj.state_counts:
            obj.state_counts[state] = int(data.get("state_counts", {}).get(state, 0))
            obj.state_cmd_error_sum[state] = float(data.get("state_cmd_error_sum", {}).get(state, 0.0))
            obj.state_speed_error_sum[state] = float(data.get("state_speed_error_sum", {}).get(state, 0.0))
        return obj

    def gate(self, args, level_gate=False):
        if level_gate:
            min_events = int(args.recovery_level_min_events_per_sign)
            overall_req = float(args.recovery_level_success_rate)
            side_req = float(args.recovery_level_side_success_rate)
        else:
            min_events = int(args.recovery_gate_min_events_per_sign)
            overall_req = float(args.recovery_gate_success_rate)
            side_req = float(args.recovery_gate_side_success_rate)

        pos = int(self.injected_pos)
        neg = int(self.injected_neg)
        spos = int(self.success_pos)
        sneg = int(self.success_neg)
        total = pos + neg
        success = spos + sneg
        overall = float(success) / total if total > 0 else 0.0
        pos_rate = float(spos) / pos if pos > 0 else 0.0
        neg_rate = float(sneg) / neg if neg > 0 else 0.0
        reasons = []
        if pos < min_events:
            reasons.append("positive injections {}<{}".format(pos, min_events))
        if neg < min_events:
            reasons.append("negative injections {}<{}".format(neg, min_events))
        if overall < overall_req:
            reasons.append("recovery success {:.1f}%<{:.1f}%".format(100*overall, 100*overall_req))
        if pos > 0 and pos_rate < side_req:
            reasons.append("positive success {:.1f}%<{:.1f}%".format(100*pos_rate, 100*side_req))
        if neg > 0 and neg_rate < side_req:
            reasons.append("negative success {:.1f}%<{:.1f}%".format(100*neg_rate, 100*side_req))
        return len(reasons) == 0, reasons, {
            "injected_pos": pos, "injected_neg": neg,
            "success_pos": spos, "success_neg": sneg,
            "overall_success_rate": overall, "positive_success_rate": pos_rate,
            "negative_success_rate": neg_rate,
        }


# =============================================================================
# Latent95 audit (diagnostic only)
# =============================================================================

class LatentAudit(object):
    def __init__(self):
        self.count = 0
        self.norm_sum = 0.0
        self.norm_max = 0.0
        self.delta_count = 0
        self.delta_sum = 0.0
        self.delta_max = 0.0
        self.sensitivity_count = 0
        self.sensitivity_sum = 0.0
        self.sensitivity_max = 0.0
        self.sensitivity_alert_count = 0
        self.failure_count = 0
        self.failure_norm_sum = 0.0
        self.failure_delta_sum = 0.0

    def update_export(self, data):
        if not isinstance(data, dict):
            return
        self.count += int(data.get("count", 0))
        self.norm_sum += float(data.get("norm_sum", 0.0))
        self.norm_max = max(self.norm_max, float(data.get("norm_max", 0.0)))
        self.delta_count += int(data.get("delta_count", 0))
        self.delta_sum += float(data.get("delta_sum", 0.0))
        self.delta_max = max(self.delta_max, float(data.get("delta_max", 0.0)))
        self.sensitivity_count += int(data.get("sensitivity_count", 0))
        self.sensitivity_sum += float(data.get("sensitivity_sum", 0.0))
        self.sensitivity_max = max(
            self.sensitivity_max, float(data.get("sensitivity_max", 0.0))
        )
        self.sensitivity_alert_count += int(data.get("sensitivity_alert_count", 0))
        self.failure_count += int(data.get("failure_count", 0))
        self.failure_norm_sum += float(data.get("failure_norm_sum", 0.0))
        self.failure_delta_sum += float(data.get("failure_delta_sum", 0.0))

    def to_dict(self):
        return {
            "count": int(self.count),
            "norm_sum": float(self.norm_sum),
            "norm_max": float(self.norm_max),
            "delta_count": int(self.delta_count),
            "delta_sum": float(self.delta_sum),
            "delta_max": float(self.delta_max),
            "sensitivity_count": int(self.sensitivity_count),
            "sensitivity_sum": float(self.sensitivity_sum),
            "sensitivity_max": float(self.sensitivity_max),
            "sensitivity_alert_count": int(self.sensitivity_alert_count),
            "failure_count": int(self.failure_count),
            "failure_norm_sum": float(self.failure_norm_sum),
            "failure_delta_sum": float(self.failure_delta_sum),
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        if isinstance(data, dict):
            obj.update_export(data)
        return obj

    def summary(self):
        return {
            "count": int(self.count),
            "norm_mean": self.norm_sum / float(self.count) if self.count else 0.0,
            "norm_max": float(self.norm_max),
            "delta_mean": self.delta_sum / float(self.delta_count) if self.delta_count else 0.0,
            "delta_max": float(self.delta_max),
            "sensitivity_mean": (
                self.sensitivity_sum / float(self.sensitivity_count)
                if self.sensitivity_count else 0.0
            ),
            "sensitivity_max": float(self.sensitivity_max),
            "sensitivity_alert_count": int(self.sensitivity_alert_count),
            "failure_count": int(self.failure_count),
            "failure_norm_mean": (
                self.failure_norm_sum / float(self.failure_count)
                if self.failure_count else 0.0
            ),
            "failure_delta_mean": (
                self.failure_delta_sum / float(self.failure_count)
                if self.failure_count else 0.0
            ),
        }


# =============================================================================
# DR exposure / coverage audit
# =============================================================================

COVERAGE_SPECS = {
    "torque_scale": DOMAIN_RANDOMIZATION_RANGES["torque_scale"],
    "max_brake_torque": DOMAIN_RANDOMIZATION_RANGES["max_brake_torque"],
    "steer_gain_positive": DOMAIN_RANDOMIZATION_RANGES["steer_gain_positive"],
    "steer_gain_negative": DOMAIN_RANDOMIZATION_RANGES["steer_gain_negative"],
    "camera_pitch_offset_deg": CAMERA_ANGLE_OFFSET_RANGES_DEG["pitch"],
    "camera_yaw_offset_deg": CAMERA_ANGLE_OFFSET_RANGES_DEG["yaw"],
    "camera_fov_offset_deg": CAMERA_FOV_OFFSET_DEG,
    "sun_altitude_angle": WEATHER_RANGES["sun_altitude_angle"],
    "sun_azimuth_angle": WEATHER_RANGES["sun_azimuth_angle"],
    "cloudiness": WEATHER_RANGES["cloudiness"],
    "wetness": WEATHER_RANGES["wetness"],
    "fog_density": WEATHER_RANGES["fog_density"],
    "brightness_gain": IMAGE_EPISODE_RANGES["brightness_gain"],
    "contrast_gain": IMAGE_EPISODE_RANGES["contrast_gain"],
    "gamma": IMAGE_EPISODE_RANGES["gamma"],
    "speed_bias_mps": STATE_SENSOR_RANGES["speed_bias_mps"],
    "speed_noise_std_mps": STATE_SENSOR_RANGES["speed_noise_std_mps"],
    "yaw_bias_rad_s": STATE_SENSOR_RANGES["yaw_bias_rad_s"],
    "yaw_noise_std_rad_s": STATE_SENSOR_RANGES["yaw_noise_std_rad_s"],
    "ax_bias_mps2": STATE_SENSOR_RANGES["ax_bias_mps2"],
    "ax_noise_std_mps2": STATE_SENSOR_RANGES["ax_noise_std_mps2"],
}


class DRExposureTracker(object):
    def __init__(self, bins=8):
        self.bins = int(bins)
        self.total_steps = 0
        self.dynamics_steps = 0
        self.vision_steps = 0
        self.sensor_steps = 0
        self.full_steps = 0
        self.map_steps = {m: 0 for m in ALL_MAPS}
        self.map_full_dr_steps = {m: 0 for m in ALL_MAPS}
        self.dr_episode_samples = 0
        self.map_dr_episodes = {m: 0 for m in ALL_MAPS}
        self.coverage = {k: [0] * self.bins for k in COVERAGE_SPECS}
        self.extra_delay_counts = {"0": 0, "1": 0}

    def to_dict(self):
        return {
            "bins": self.bins,
            "total_steps": self.total_steps,
            "dynamics_steps": self.dynamics_steps,
            "vision_steps": self.vision_steps,
            "sensor_steps": self.sensor_steps,
            "full_steps": self.full_steps,
            "map_steps": dict(self.map_steps),
            "map_full_dr_steps": dict(self.map_full_dr_steps),
            "dr_episode_samples": self.dr_episode_samples,
            "map_dr_episodes": dict(self.map_dr_episodes),
            "coverage": copy.deepcopy(self.coverage),
            "extra_delay_counts": dict(self.extra_delay_counts),
        }

    @classmethod
    def from_dict(cls, data, default_bins=8):
        obj = cls(int(data.get("bins", default_bins)) if isinstance(data, dict) else default_bins)
        if not isinstance(data, dict):
            return obj
        for name in ("total_steps", "dynamics_steps", "vision_steps", "sensor_steps", "full_steps", "dr_episode_samples"):
            setattr(obj, name, int(data.get(name, 0)))
        for m in ALL_MAPS:
            obj.map_steps[m] = int(data.get("map_steps", {}).get(m, 0))
            obj.map_full_dr_steps[m] = int(data.get("map_full_dr_steps", {}).get(m, 0))
            obj.map_dr_episodes[m] = int(data.get("map_dr_episodes", {}).get(m, 0))
        for key in obj.coverage:
            vals = data.get("coverage", {}).get(key)
            if isinstance(vals, list) and len(vals) == obj.bins:
                obj.coverage[key] = [int(x) for x in vals]
        obj.extra_delay_counts.update({str(k): int(v) for k, v in data.get("extra_delay_counts", {}).items()})
        return obj

    def add_rollout(self, map_path, samples, dynamics, vision, sensor):
        samples = int(samples)
        self.total_steps += samples
        self.map_steps[map_path] = int(self.map_steps.get(map_path, 0)) + samples
        if dynamics:
            self.dynamics_steps += samples
        if vision:
            self.vision_steps += samples
        if sensor:
            self.sensor_steps += samples
        if dynamics and vision and sensor:
            self.full_steps += samples
            self.map_full_dr_steps[map_path] = int(self.map_full_dr_steps.get(map_path, 0)) + samples

    def _bin_index(self, key, value):
        low, high = COVERAGE_SPECS[key]
        if high <= low:
            return 0
        ratio = (float(value) - float(low)) / (float(high) - float(low))
        ratio = max(0.0, min(0.999999999, ratio))
        return int(ratio * self.bins)

    def add_episode_sample(self, sample):
        if not isinstance(sample, dict) or not bool(sample.get("full_dr", False)):
            return
        map_path = str(sample.get("map_path"))
        dyn = sample.get("domain", {})
        vis = sample.get("vision", {})
        self.dr_episode_samples += 1
        if map_path in self.map_dr_episodes:
            self.map_dr_episodes[map_path] += 1
        merged = {}
        merged.update(dyn if isinstance(dyn, dict) else {})
        merged.update(vis if isinstance(vis, dict) else {})
        for key in self.coverage:
            if key in merged:
                idx = self._bin_index(key, merged[key])
                self.coverage[key][idx] += 1
        delay = str(int(dyn.get("extra_command_delay_ticks", 0))) if isinstance(dyn, dict) else "0"
        self.extra_delay_counts[delay] = int(self.extra_delay_counts.get(delay, 0)) + 1

    def coverage_fractions(self):
        return {
            key: float(sum(1 for x in counts if int(x) > 0)) / float(len(counts))
            for key, counts in self.coverage.items()
        }

    def gate(self, args):
        reasons = []
        ratio = float(self.full_steps) / float(max(self.total_steps, 1))
        if self.full_steps < int(args.dr_min_full_steps):
            reasons.append("full DR steps {}<{}".format(self.full_steps, args.dr_min_full_steps))
        if ratio < float(args.dr_min_full_ratio):
            reasons.append("full DR ratio {:.1f}%<{:.1f}%".format(100*ratio, 100*args.dr_min_full_ratio))
        if self.dr_episode_samples < int(args.dr_min_episode_samples):
            reasons.append("DR episode samples {}<{}".format(self.dr_episode_samples, args.dr_min_episode_samples))
        for map_path in ALL_MAPS:
            if int(self.map_dr_episodes.get(map_path, 0)) < int(args.dr_min_episodes_per_map):
                reasons.append("{} DR episodes {}<{}".format(
                    map_short(map_path), self.map_dr_episodes.get(map_path, 0), args.dr_min_episodes_per_map
                ))
        fractions = self.coverage_fractions()
        low_keys = [k for k, v in fractions.items() if v < float(args.dr_coverage_min_fraction)]
        if low_keys:
            reasons.append("DR bin coverage low: {}".format(
                ", ".join("{}={:.0f}%".format(k, 100*fractions[k]) for k in low_keys)
            ))
        # Timing DR must have both 0 and 1 tick represented.
        if int(self.extra_delay_counts.get("0", 0)) <= 0 or int(self.extra_delay_counts.get("1", 0)) <= 0:
            reasons.append("extra_command_delay_ticks coverage missing 0 or 1")
        return len(reasons) == 0, reasons, {
            "total_steps": int(self.total_steps),
            "full_steps": int(self.full_steps),
            "full_ratio": float(ratio),
            "dr_episode_samples": int(self.dr_episode_samples),
            "coverage_fraction": fractions,
            "map_dr_episodes": dict(self.map_dr_episodes),
            "extra_delay_counts": dict(self.extra_delay_counts),
        }


# =============================================================================
# Failure clustering
# =============================================================================

class FailureClusterTracker(object):
    def __init__(self, grid_size=1.5):
        self.grid_size = float(grid_size)
        self.cells = {}
        self.map_offroads = {m: 0 for m in ALL_MAPS}

    def _key(self, event):
        gx = int(round(float(event["world_x"]) / self.grid_size))
        gy = int(round(float(event["world_y"]) / self.grid_size))
        return "{}|{}|{}|{}".format(event["map_path"], int(event.get("spawn_number", 0)), gx, gy)

    def add(self, event):
        if not isinstance(event, dict) or str(event.get("reason")) != "offroad":
            return
        map_path = str(event.get("map_path"))
        self.map_offroads[map_path] = int(self.map_offroads.get(map_path, 0)) + 1
        key = self._key(event)
        rec = self.cells.setdefault(key, {
            "map_path": map_path,
            "spawn_number": int(event.get("spawn_number", 0)),
            "count": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_progress": 0.0,
        })
        rec["count"] += 1
        rec["sum_x"] += float(event["world_x"])
        rec["sum_y"] += float(event["world_y"])
        rec["sum_progress"] += float(event.get("episode_progress_m", 0.0))

    def alert_maps(self, args):
        result = set()
        for rec in self.cells.values():
            if int(rec.get("count", 0)) >= int(args.failure_cluster_alert_count):
                result.add(str(rec["map_path"]))
        return result

    def top_clusters(self, limit=10):
        items = sorted(self.cells.values(), key=lambda x: int(x.get("count", 0)), reverse=True)
        out = []
        for rec in items[:int(limit)]:
            n = max(int(rec["count"]), 1)
            item = dict(rec)
            item["mean_x"] = float(rec["sum_x"]) / n
            item["mean_y"] = float(rec["sum_y"]) / n
            item["mean_progress_m"] = float(rec["sum_progress"]) / n
            out.append(item)
        return out

    def to_dict(self):
        return {
            "grid_size": self.grid_size,
            "cells": copy.deepcopy(self.cells),
            "map_offroads": dict(self.map_offroads),
        }

    @classmethod
    def from_dict(cls, data, default_grid=1.5):
        obj = cls(float(data.get("grid_size", default_grid)) if isinstance(data, dict) else default_grid)
        if isinstance(data, dict):
            if isinstance(data.get("cells"), dict):
                obj.cells = copy.deepcopy(data["cells"])
            for m in ALL_MAPS:
                obj.map_offroads[m] = int(data.get("map_offroads", {}).get(m, 0))
        return obj


# =============================================================================
# Checkpoint/state management
# =============================================================================

def checkpoint_dir():
    path = os.path.join(PPO_CHECKPOINT_DIR, MODEL_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def state_path():
    return os.path.join(checkpoint_dir(), STATE_NAME)


def champion_path(kind):
    return os.path.join(checkpoint_dir(), CHAMPION_FILES[str(kind)])


def numbered_policy_files():
    pattern = re.compile(r"^ppo_policy_(\d+)_\.pth$")
    result = []
    for name in os.listdir(checkpoint_dir()):
        match = pattern.match(name)
        if match:
            result.append((int(match.group(1)), os.path.join(checkpoint_dir(), name)))
    result.sort(key=lambda x: x[0])
    return result


def _v5_payload(agent):
    return {
        "version": "PPO_V5_CLEAN",
        "obs_dim": int(agent.obs_dim),
        "action_dim": int(agent.action_dim),
        "action_std": float(agent.action_std),
        "policy_state_dict": copy.deepcopy(agent.old_policy.state_dict()),
        "optimizer_state_dict": copy.deepcopy(agent.optimizer.state_dict()),
    }


def save_category_champion(kind, agent, global_step, stage, checkpoint_path_value, verification):
    """
    Save every category champion in the SAME PPO_V5_CLEAN format used by
    PPOAgent.load()/Jetson inference.  Extra VNext metadata is additive only;
    older loaders ignore it safely.
    """
    kind = str(kind)
    payload = _v5_payload(agent)
    payload["vnext_champion_kind"] = kind
    payload["vnext_metadata"] = {
        "trainer_version": TRAINER_VERSION,
        "global_step": int(global_step),
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "checkpoint_path": str(checkpoint_path_value),
        "verification": copy.deepcopy(verification),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, champion_path(kind))

    sidecar = champion_path(kind) + ".json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(payload["vnext_metadata"], handle, indent=2, sort_keys=True)
    return champion_path(kind)


def load_category_champion(kind, agent):
    path = champion_path(kind)
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location=agent.device)
    if payload.get("version") != "PPO_V5_CLEAN":
        raise RuntimeError("Champion is not PPO_V5_CLEAN-loadable: {}".format(path))
    agent.policy.load_state_dict(payload["policy_state_dict"], strict=True)
    agent.old_policy.load_state_dict(payload["policy_state_dict"], strict=True)
    if "optimizer_state_dict" in payload:
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.set_action_std(float(payload.get("action_std", agent.action_std)))
    agent.memory.clear()
    meta = dict(payload.get("vnext_metadata", {}))
    meta["checkpoint_path"] = str(meta.get("checkpoint_path") or path)
    meta["global_step"] = int(meta.get("global_step", 0))
    return meta

def _champion_saved_step(kind):
    path = champion_path(kind)
    if not os.path.isfile(path):
        return -1
    sidecar = path + ".json"
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as handle:
                return int(json.load(handle).get("global_step", -1))
        except Exception:
            pass
    try:
        payload = torch.load(path, map_location="cpu")
        return int(payload.get("vnext_metadata", {}).get("global_step", -1))
    except Exception:
        return -1


def highest_available_champion(stage):
    stage = int(stage)
    if stage == 0:
        candidates = ["clean"]
    elif stage == 1:
        candidates = ["recovery", "clean"]
    elif stage == 2:
        # Critical: do NOT blindly prefer a very old multimap champion.
        # Roll back to the newest safe Stage2-compatible checkpoint so one
        # canary failure cannot erase >1M transitions of useful learning.
        candidates = ["multimap", "recovery", "clean"]
        existing = [k for k in candidates if os.path.isfile(champion_path(k))]
        return max(existing, key=_champion_saved_step) if existing else None
    else:
        # A verified Sim2Real champion keeps semantic priority in Stage3.
        if os.path.isfile(champion_path("sim2real")):
            return "sim2real"
        candidates = ["multimap", "recovery", "clean"]
        existing = [k for k in candidates if os.path.isfile(champion_path(k))]
        return max(existing, key=_champion_saved_step) if existing else None

    for kind in candidates:
        if os.path.isfile(champion_path(kind)):
            return kind
    return None


def save_numbered_checkpoint(agent, global_step, evaluated_stage, verification):
    path = agent.save()
    sidecar = path + ".vnext.json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump({
            "gate": "CLEAN_ANCHOR_PASS",
            "global_step": int(global_step),
            "evaluated_stage": int(evaluated_stage),
            "evaluated_stage_name": STAGE_NAMES[int(evaluated_stage)],
            "verification": copy.deepcopy(verification),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }, handle, indent=2, sort_keys=True)
    print("VERIFIED POLICY FILE | {} | sidecar={}".format(path, sidecar))
    return path


def _prune(directory, prefix, keep):
    if not os.path.isdir(directory):
        return
    items = []
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".pth"):
            full = os.path.join(directory, name)
            items.append((os.path.getmtime(full), full))
    items.sort(reverse=True)
    for _, full in items[max(0, int(keep)):]:
        try:
            os.remove(full)
        except Exception:
            pass


def save_rejected(agent, global_step, stage, reasons, keep):
    directory = os.path.join(checkpoint_dir(), "rejected")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "rejected_step{}_stage{}_{}.pth".format(
        int(global_step), int(stage), datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    payload = _v5_payload(agent)
    payload["vnext_kind"] = "ANCHOR_CANARY_REJECTED"
    payload["metadata"] = {"reasons": list(reasons), "global_step": int(global_step), "stage": int(stage)}
    torch.save(payload, path)
    _prune(directory, "rejected_", keep)
    return path


def save_emergency(agent, global_step, stage, reason, keep):
    directory = os.path.join(checkpoint_dir(), "emergency")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "emergency_step{}_stage{}_{}.pth".format(
        int(global_step), int(stage), datetime.now().strftime("%Y%m%d_%H%M%S")
    ))
    payload = _v5_payload(agent)
    payload["vnext_kind"] = "EMERGENCY_UNVERIFIED"
    payload["metadata"] = {"reason": str(reason), "global_step": int(global_step), "stage": int(stage)}
    torch.save(payload, path)
    _prune(directory, "emergency_", keep)
    return path


def write_state(agent, global_step, episode, stage, stage_start_step,
                consecutive_anchor_passes, current_checkpoint, mastery,
                exposure, failure_clusters, recovery_audit, latent_audit, recovery_level,
                stage3_fail_streak, repair_until_step, args):
    data = {
        "version": TRAINER_VERSION,
        "model_name": MODEL_NAME,
        "global_step": int(global_step),
        "episode": int(episode),
        "stage": int(stage),
        "stage_name": STAGE_NAMES[int(stage)],
        "stage_start_step": int(stage_start_step),
        "consecutive_anchor_passes": int(consecutive_anchor_passes),
        "action_std": float(agent.action_std),
        "checkpoint_path": str(current_checkpoint) if current_checkpoint else None,
        "mastery": mastery.to_dict(),
        "dr_exposure": exposure.to_dict(),
        "failure_clusters": failure_clusters.to_dict(),
        "recovery_audit": recovery_audit.to_dict(),
        "latent_audit": latent_audit.to_dict(),
        "recovery_level": int(recovery_level),
        "stage3_fail_streak": int(stage3_fail_streak),
        "repair_until_step": int(repair_until_step),
        "category_champions": {
            kind: champion_path(kind) if os.path.isfile(champion_path(kind)) else None
            for kind in CHAMPION_FILES
        },
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "safe_spawns": list(args.safe_spawns),
    }
    with open(state_path(), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return state_path()


def read_state():
    with open(state_path(), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    state_version = data.get("version")
    if state_version != TRAINER_VERSION and state_version not in LEGACY_STATE_VERSIONS:
        raise RuntimeError("Training state version mismatch: {}".format(state_version))
    if state_version != TRAINER_VERSION:
        print("STATE MIGRATION | {} -> {} | scheduler state is recomputed safely".format(
            state_version, TRAINER_VERSION
        ))
    return data


def prepare_fresh_or_resume(agent, args):
    if bool(args.resume):
        if not os.path.isfile(state_path()):
            raise FileNotFoundError(state_path())
        data = read_state()
        cp = str(data.get("checkpoint_path") or "")
        if not cp or not os.path.isfile(cp):
            raise FileNotFoundError("Resume checkpoint missing: {}".format(cp))
        loaded = agent.load(checkpoint_path=cp)
        resume_override = str(getattr(args, "resume_policy_override", "") or "").strip()
        if resume_override:
            if not os.path.isfile(resume_override):
                raise FileNotFoundError("Resume policy override missing: {}".format(resume_override))
            loaded = agent.load(checkpoint_path=resume_override)
            cp = resume_override
            agent.memory.clear()
            print("RESUME POLICY OVERRIDE | weights={} | state_step={} stage={}".format(
                resume_override, int(data.get("global_step", 0)), STAGE_NAMES[int(data.get("stage", 0))]
            ))
        if "action_std" in data:
            agent.set_action_std(float(data["action_std"]))
        return {
            "global_step": int(data.get("global_step", 0)),
            "episode": int(data.get("episode", 0)),
            "stage": int(data.get("stage", 0)),
            "stage_start_step": int(data.get("stage_start_step", 0)),
            "consecutive_anchor_passes": int(data.get("consecutive_anchor_passes", 0)),
            "loaded_checkpoint": loaded,
            "current_checkpoint": cp,
            "mastery": MapMasteryTracker.from_dict(data.get("mastery", {})),
            "exposure": DRExposureTracker.from_dict(data.get("dr_exposure", {}), args.dr_coverage_bins),
            "failure_clusters": FailureClusterTracker.from_dict(data.get("failure_clusters", {}), args.failure_cluster_grid_carla_m),
            "recovery_audit": RecoveryAudit.from_dict(data.get("recovery_audit", {})),
            "latent_audit": LatentAudit.from_dict(data.get("latent_audit", {})),
            "recovery_level": int(data.get("recovery_level", 1)),
            "stage3_fail_streak": int(data.get("stage3_fail_streak", 0)),
            "repair_until_step": int(data.get("repair_until_step", 0)),
        }

    # New VNext run must use a new/empty model directory.
    if numbered_policy_files() or os.path.isfile(state_path()) or any(
        os.path.isfile(champion_path(k)) for k in CHAMPION_FILES
    ):
        raise RuntimeError(
            "FRESH SAFETY STOP: '{}' already contains VNext state/checkpoints. "
            "Archive it or use --resume true.".format(checkpoint_dir())
        )

    loaded = None
    init_checkpoint = str(args.init_checkpoint).strip()
    if init_checkpoint:
        if not os.path.isfile(init_checkpoint):
            raise FileNotFoundError(init_checkpoint)
        loaded = agent.load(checkpoint_path=init_checkpoint)
        agent.memory.clear()

    stage = int(args.init_stage) if loaded else 0
    return {
        "global_step": 0,
        "episode": 0,
        "stage": stage,
        "stage_start_step": 0,
        "consecutive_anchor_passes": 0,
        "loaded_checkpoint": loaded,
        "current_checkpoint": loaded,
        "mastery": MapMasteryTracker(),
        "exposure": DRExposureTracker(args.dr_coverage_bins),
        "failure_clusters": FailureClusterTracker(args.failure_cluster_grid_carla_m),
        "recovery_audit": RecoveryAudit(),
        "latent_audit": LatentAudit(),
        "recovery_level": 1,
        "stage3_fail_streak": 0,
        "repair_until_step": 0,
    }


# =============================================================================
# Worker runtime
# =============================================================================

class WorkerRuntime(object):
    def __init__(self, worker_id, cfg):
        self.worker_id = int(worker_id)
        self.cfg = dict(cfg)
        torch.set_num_threads(int(cfg["worker_torch_threads"]))
        seed_everything(int(cfg["seed"]))
        self.client = carla.Client(str(cfg["host"]), int(cfg["port"]))
        self.client.set_timeout(float(cfg["carla_timeout"]))
        self.agent = PPOAgent(
            town="__vnext_worker_{}".format(worker_id),
            action_std_init=float(cfg["action_std_init"]),
            device=cfg["worker_ppo_device"],
        )
        self.env = None
        self.observation = None
        self.current_map = None
        self.current_signature = None
        self.episode_stats = EpisodeAccumulator()
        self.completed_episode_count = 0
        self.pending_recovery_sign = None
        self.episode_progress_m = 0.0
        self.last_dr_signature = None
        self.prev_latent = None
        self.last_latent_sensitivity = 0.0

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
        self.pending_recovery_sign = None
        self.episode_progress_m = 0.0
        self.last_dr_signature = None
        self.prev_latent = None
        self.last_latent_sensitivity = 0.0

    def close(self):
        self.close_env()

    def available_maps(self):
        try:
            return list(self.client.get_available_maps())
        except Exception as exc:
            raise RuntimeError("Worker {} could not query CARLA available maps: {}".format(
                self.worker_id, exc
            ))

    def _resolve_configured_map(self, map_path):
        available = self.available_maps()
        resolved, hints = resolve_map_path_from_available(map_path, available)
        if resolved is None:
            hint_text = "\n    ".join(hints) if hints else "<no similar map names found>"
            raise RuntimeError(
                "CARLA MAP NOT FOUND / UNRESOLVED\n"
                "  worker={} port={}\n"
                "  requested={}\n"
                "  similar available maps:\n    {}\n"
                "Use client.get_available_maps() and correct AUX_OVAL_MAPS; "
                "VNext will not silently skip a required map.".format(
                    self.worker_id, self.cfg["port"], map_path, hint_text
                )
            )
        return str(resolved)

    def _load_world(self, map_path):
        requested = str(map_path)
        resolved = requested
        try:
            world = self.client.load_world(requested)
        except RuntimeError as exc:
            if "map not found" not in str(exc).lower():
                raise
            resolved = self._resolve_configured_map(requested)
            if resolved.replace("\\", "/").lower() != requested.replace("\\", "/").lower():
                print("MAP PATH AUTO-RESOLVE | worker={} | requested={} | actual={}".format(
                    self.worker_id, requested, resolved
                ))
            world = self.client.load_world(resolved)

        actual = str(world.get_map().name).replace("\\", "/").lower()
        resolved_tail = str(resolved).replace("\\", "/").lower().split("/")[-1]
        if resolved_tail and resolved_tail not in actual:
            raise RuntimeError("Worker {} map mismatch: requested={} resolved={} loaded={}".format(
                self.worker_id, requested, resolved, world.get_map().name
            ))
        world.set_weather(carla.WeatherParameters.CloudyNoon)
        return world

    def _configure_recovery(self, command):
        configure_recovery_scenarios_v5(
            recovery_spawn_probability=0.0,
            spawn_lateral_min_m=0.0,
            spawn_lateral_max_m=0.0,
            spawn_heading_min_deg=0.0,
            spawn_heading_max_deg=0.0,
            disturbance_probability=float(command.get("recovery_prob", 0.0)),
            disturbance_duration_min_s=float(command.get("recovery_duration_min_s", self.cfg["recovery_duration_min_s"])),
            disturbance_duration_max_s=float(command.get("recovery_duration_max_s", self.cfg["recovery_duration_max_s"])),
            disturbance_steer_min=float(command.get("recovery_steer_min", self.cfg["recovery_steer_min"])),
            disturbance_steer_max=float(command.get("recovery_steer_max", self.cfg["recovery_steer_max"])),
            recovery_min_episode_step=int(self.cfg["recovery_min_episode_step"]),
            recovery_stable_ticks=int(self.cfg["recovery_stable_ticks"]),
            recovery_stable_lateral_m=float(self.cfg["recovery_stable_lateral_m"]),
            recovery_stable_heading_deg=float(self.cfg["recovery_stable_heading_deg"]),
            recovery_stable_speed_mps=float(self.cfg["recovery_stable_speed_mps"]),
            recovery_cooldown_ticks=int(self.cfg["recovery_cooldown_ticks"]),
            recovery_max_events_per_episode=int(self.cfg["recovery_max_events"]),
            seed=int(command.get("scenario_seed", self.cfg["seed"])),
        )

    def ensure_train_env(self, command):
        signature = (
            str(command["map_path"]), bool(command["dynamics_dr"]),
            bool(command["vision_dr"]), bool(command["sensor_dr"]),
            bool(command["recovery"]), round(float(command["recovery_prob"]), 6),
            round(float(command.get("recovery_steer_min", 0.0)), 4),
            round(float(command.get("recovery_steer_max", 0.0)), 4),
            tuple(int(x) for x in command["safe_spawns"]), int(command["scenario_seed"]),
        )
        if self.env is not None and signature == self.current_signature:
            return
        self.close_env()
        self._configure_recovery(command)
        world = self._load_world(command["map_path"])
        self.env = CarlaEnvironmentRGBV5MultiMap(
            client=self.client, world=world,
            town="vnext_worker_{}".format(self.worker_id),
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
        self.pending_recovery_sign = None
        self.episode_progress_m = 0.0
        self.last_dr_signature = None
        self.prev_latent = None
        self.last_latent_sensitivity = 0.0

    @staticmethod
    def _dr_signature(info):
        dyn = info.get("domain_randomization", {})
        vis = info.get("vision_sensor_dr", {})
        if not isinstance(dyn, dict) or not isinstance(vis, dict):
            return None
        return (
            round(float(dyn.get("torque_scale", 0.0)), 6),
            round(float(dyn.get("steer_gain_positive", 0.0)), 6),
            round(float(dyn.get("steer_gain_negative", 0.0)), 6),
            int(dyn.get("extra_command_delay_ticks", 0)),
            int(vis.get("episode_noise_seed", 0)),
        )

    @staticmethod
    def _latent95_numpy(observation):
        if isinstance(observation, torch.Tensor):
            array = observation.detach().cpu().numpy()
        else:
            array = np.asarray(observation)
        array = np.asarray(array, dtype=np.float32).reshape(-1)
        if array.size < 95:
            raise RuntimeError("Observation shorter than latent95")
        return np.asarray(array[:95], dtype=np.float32)

    def _probe_latent_sensitivity(self, observation, local_i):
        """Finite-difference steer sensitivity; NEVER used as the train action."""
        eps = float(self.cfg["latent_sensitivity_epsilon_l2"])
        obs_tensor = self.agent._obs_tensor(observation)
        with torch.no_grad():
            base_action = self.agent.old_policy.deterministic_action(obs_tensor)

            rng = np.random.RandomState(
                int(self.cfg["seed"])
                + int(self.completed_episode_count) * 100003
                + int(local_i) * 97
            )
            direction = rng.normal(0.0, 1.0, size=(95,)).astype(np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                return 0.0
            direction /= norm

            perturbed = obs_tensor.clone()
            delta = torch.as_tensor(
                direction * eps,
                dtype=torch.float32,
                device=perturbed.device,
            )
            perturbed[:95] = perturbed[:95] + delta
            perturbed_action = self.agent.old_policy.deterministic_action(perturbed)

            steer_delta = abs(
                float(perturbed_action[0].detach().cpu())
                - float(base_action[0].detach().cpu())
            )
        return float(steer_delta / eps)

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
        failure_events = []
        dr_episode_samples = []
        audit = {
            "injected_pos": 0, "injected_neg": 0,
            "success_pos": 0, "success_neg": 0,
            "state_counts": {s: 0 for s in ("stable", "mild", "moderate", "strong")},
            "state_cmd_error_sum": {s: 0.0 for s in ("stable", "mild", "moderate", "strong")},
            "state_speed_error_sum": {s: 0.0 for s in ("stable", "mild", "moderate", "strong")},
        }
        latent_export = {
            "count": 0, "norm_sum": 0.0, "norm_max": 0.0,
            "delta_count": 0, "delta_sum": 0.0, "delta_max": 0.0,
            "sensitivity_count": 0, "sensitivity_sum": 0.0,
            "sensitivity_max": 0.0, "sensitivity_alert_count": 0,
            "failure_count": 0, "failure_norm_sum": 0.0,
            "failure_delta_sum": 0.0,
        }
        start = time.time()
        artificial_cut = False

        for local_i in range(requested_steps):
            latent_norm = 0.0
            latent_delta = 0.0
            if bool(self.cfg.get("latent_audit", True)):
                latent = self._latent95_numpy(self.observation)
                latent_norm = float(np.linalg.norm(latent))
                latent_export["count"] += 1
                latent_export["norm_sum"] += latent_norm
                latent_export["norm_max"] = max(latent_export["norm_max"], latent_norm)

                if self.prev_latent is not None:
                    latent_delta = float(np.linalg.norm(latent - self.prev_latent))
                    latent_export["delta_count"] += 1
                    latent_export["delta_sum"] += latent_delta
                    latent_export["delta_max"] = max(latent_export["delta_max"], latent_delta)
                self.prev_latent = latent.copy()

                if local_i % int(self.cfg["latent_sensitivity_every_steps"]) == 0:
                    sensitivity = self._probe_latent_sensitivity(self.observation, local_i)
                    self.last_latent_sensitivity = float(sensitivity)
                    latent_export["sensitivity_count"] += 1
                    latent_export["sensitivity_sum"] += float(sensitivity)
                    latent_export["sensitivity_max"] = max(
                        latent_export["sensitivity_max"], float(sensitivity)
                    )
                    if sensitivity >= float(self.cfg["latent_sensitivity_alert"]):
                        latent_export["sensitivity_alert_count"] += 1

            action = self.agent.get_action(self.observation, train=True)
            next_obs, reward, done, info = self.env.step(action)

            actual_terminated = bool(info.get("terminated", False))
            actual_truncated = bool(info.get("truncated", False))
            is_last = local_i == requested_steps - 1
            artificial_cut = bool(is_last and not actual_terminated and not actual_truncated)
            train_truncated = bool(actual_truncated or artificial_cut)
            self.agent.record_outcome(
                reward=reward, terminated=actual_terminated, truncated=train_truncated,
                next_obs=(next_obs if train_truncated else None),
            )

            self.episode_stats.add(action=action, reward=reward, info=info)
            diagnostics.add(reward=reward, info=info)
            self.episode_progress_m += float(info.get("forward_progress_m", 0.0))

            # State-speed audit.
            state = str(info.get("recovery_state", ""))
            if state in audit["state_counts"]:
                target = float(info.get("adaptive_target_speed_mps", 0.0))
                cmd = float(info.get("speed_cmd_mps", action[1]))
                speed = float(info.get("speed_mps", 0.0))
                audit["state_counts"][state] += 1
                audit["state_cmd_error_sum"][state] += abs(cmd - target)
                audit["state_speed_error_sum"][state] += abs(speed - target)

            # Pair injected sign with later success event, even across rollout cuts.
            if bool(info.get("disturbance_trigger_event", False)):
                steer = float(info.get("disturbance_steer", 0.0))
                self.pending_recovery_sign = 1 if steer > 0 else (-1 if steer < 0 else None)
                if self.pending_recovery_sign == 1:
                    audit["injected_pos"] += 1
                elif self.pending_recovery_sign == -1:
                    audit["injected_neg"] += 1
            if bool(info.get("disturbance_success_event", False)):
                if self.pending_recovery_sign == 1:
                    audit["success_pos"] += 1
                elif self.pending_recovery_sign == -1:
                    audit["success_neg"] += 1
                self.pending_recovery_sign = None

            # Episode-level DR sample: count once whenever sampled params change.
            sig = self._dr_signature(info)
            if sig is not None and sig != self.last_dr_signature:
                self.last_dr_signature = sig
                dr_episode_samples.append({
                    "map_path": self.current_map,
                    "spawn_number": info.get("spawn_number"),
                    "full_dr": bool(info.get("dynamics_dr_enabled", False)
                                    and info.get("vision_dr_enabled", False)
                                    and info.get("sensor_dr_enabled", False)),
                    "domain": dict(info.get("domain_randomization", {})),
                    "vision": dict(info.get("vision_sensor_dr", {})),
                })

            if done:
                self.completed_episode_count += 1
                summary = self.episode_stats.finish(info)
                summary["map_path"] = self.current_map
                summary["spawn_number"] = info.get("spawn_number")
                completed.append(summary)

                reason = str(info.get("termination_reason"))
                if reason in ("offroad", "collision", "stuck"):
                    transform = self.env.vehicle.get_transform()
                    failure_events.append({
                        "map_path": self.current_map,
                        "spawn_number": int(info.get("spawn_number") or 0),
                        "reason": reason,
                        "world_x": float(transform.location.x),
                        "world_y": float(transform.location.y),
                        "world_yaw_deg": float(transform.rotation.yaw),
                        "episode_progress_m": float(self.episode_progress_m),
                        "lateral_error_m": float(info.get("lateral_error_m", 0.0)),
                        "heading_error_deg": math.degrees(float(info.get("heading_error_rad", 0.0))),
                        "camera_frame_lag": info.get("camera_frame_lag"),
                        "recovery_state": str(info.get("recovery_state", "")),
                        "latent_norm": float(latent_norm),
                        "latent_delta_l2": float(latent_delta),
                        "latent_steer_sensitivity": float(self.last_latent_sensitivity),
                    })
                    if bool(self.cfg.get("latent_audit", True)):
                        latent_export["failure_count"] += 1
                        latent_export["failure_norm_sum"] += float(latent_norm)
                        latent_export["failure_delta_sum"] += float(latent_delta)

                self.observation = self.env.reset()
                self.episode_stats = EpisodeAccumulator()
                self.episode_progress_m = 0.0
                self.pending_recovery_sign = None
                self.last_dr_signature = None
                self.prev_latent = None
                self.last_latent_sensitivity = 0.0
            else:
                self.observation = next_obs

        rollout = export_memory(self.agent.memory)
        self.agent.memory.clear()
        return {
            "type": "rollout", "worker": self.worker_id,
            "map_path": self.current_map, "samples": requested_steps,
            "wall_s": float(time.time() - start), "rollout": rollout,
            "diagnostics": diagnostics.export(), "episodes": completed,
            "worker_episode_count": self.completed_episode_count,
            "artificial_final_cut": bool(artificial_cut),
            "recovery_audit": audit,
            "latent_audit": latent_export,
            "failure_events": failure_events,
            "dr_episode_samples": dr_episode_samples,
            "dynamics_dr": bool(command["dynamics_dr"]),
            "vision_dr": bool(command["vision_dr"]),
            "sensor_dr": bool(command["sensor_dr"]),
        }

    def evaluate_policy(self, command):
        """Generic deterministic evaluator for clean mastery and robust DR gates."""
        self.close_env()
        self._configure_recovery({
            "recovery_prob": float(command.get("recovery_prob", 0.0)),
            "scenario_seed": int(command["scenario_seed"]),
            "recovery_steer_min": float(command.get("recovery_steer_min", 0.10)),
            "recovery_steer_max": float(command.get("recovery_steer_max", 0.18)),
            "recovery_duration_min_s": float(command.get("recovery_duration_min_s", 0.10)),
            "recovery_duration_max_s": float(command.get("recovery_duration_max_s", 0.16)),
        })
        world = self._load_world(command["map_path"])
        env = CarlaEnvironmentRGBV5MultiMap(
            client=self.client, world=world,
            town="__vnext_eval_worker_{}".format(self.worker_id),
            safe_spawn_numbers=list(command["safe_spawns"]),
            desired_speed_mps=1.0,
            max_episode_seconds=float(command["max_steps"]) / CONTROL_HZ + 5.0,
            encoder_device=self.cfg["encoder_device"],
            dynamics_dr_enabled=bool(command.get("dynamics_dr", False)),
            vision_dr_enabled=bool(command.get("vision_dr", False)),
            sensor_dr_enabled=bool(command.get("sensor_dr", False)),
            domain_randomization_seed=int(command["dr_seed"]),
            recovery_scenarios_enabled=bool(command.get("recovery", False)),
        )
        load_numpy_state(self.agent.policy, command["policy_state"])
        load_numpy_state(self.agent.old_policy, command["policy_state"])
        self.agent.memory.clear()

        results = []
        repeats = int(command.get("repeats", 1))
        try:
            for spawn_number in command["safe_spawns"]:
                for repeat_index in range(repeats):
                    env.force_next_spawn_number(int(spawn_number))
                    obs = env.reset()
                    n = 0
                    progress = 0.0
                    reward_sum = 0.0
                    abs_lat_sum = 0.0
                    max_lat = 0.0
                    abs_heading_sum = 0.0
                    max_heading = 0.0
                    abs_steer_sum = 0.0
                    steer075 = 0
                    steer050 = 0
                    edge_ticks = 0
                    center_escape = 0
                    initial_rho = None
                    reason = "eval_horizon"
                    last_info = {}

                    for step_idx in range(int(command["max_steps"])):
                        action = self.agent.get_action(obs, train=False)
                        next_obs, reward, done, info = env.step(action)
                        last_info = info
                        n += 1
                        reward_sum += float(reward)
                        progress += float(info.get("forward_progress_m", 0.0))
                        lat = abs(float(info.get("lateral_error_m", 0.0)))
                        head = abs(float(info.get("heading_error_rad", 0.0)))
                        abs_lat_sum += lat
                        max_lat = max(max_lat, lat)
                        abs_heading_sum += head
                        max_heading = max(max_heading, head)
                        control = info.get("control", {})
                        steer = float(control.get("actual_steer_cmd", action[0])) if isinstance(control, dict) else float(action[0])
                        abs_steer_sum += abs(steer)
                        if abs(steer) >= 0.50:
                            steer050 += 1
                        if abs(steer) >= 0.75:
                            steer075 += 1

                        try:
                            wp = env.reward_manager.road_metrics._driving_waypoint_projected()
                            half_lane_real = float(wp.lane_width) / float(CARLA_GEOMETRY_SCALE) / 2.0 if wp is not None else None
                        except Exception:
                            half_lane_real = None
                        if half_lane_real is not None and half_lane_real > 1e-6:
                            rho = lat / half_lane_real
                            if initial_rho is None:
                                initial_rho = rho
                            if rho >= float(command["edge_ratio_threshold"]):
                                edge_ticks += 1
                            if (
                                step_idx < int(command["center_escape_window_ticks"])
                                and initial_rho <= float(command["center_start_ratio"])
                                and rho >= float(command["center_escape_ratio"])
                            ):
                                center_escape = 1
                        obs = next_obs
                        if done:
                            reason = str(info.get("termination_reason"))
                            break

                    denom = float(max(n, 1))
                    results.append({
                        "worker": self.worker_id,
                        "map_path": str(command["map_path"]),
                        "spawn_number": int(spawn_number),
                        "repeat_index": int(repeat_index),
                        "steps": int(n), "reason": reason,
                        "reward": float(reward_sum), "progress_m": float(progress),
                        "avg_abs_lateral_m": float(abs_lat_sum / denom),
                        "max_abs_lateral_m": float(max_lat),
                        "avg_abs_heading_deg": float(math.degrees(abs_heading_sum / denom)),
                        "max_abs_heading_deg": float(math.degrees(max_heading)),
                        "avg_abs_steer": float(abs_steer_sum / denom),
                        "steer_gt_050_rate": float(steer050 / denom),
                        "steer_gt_075_rate": float(steer075 / denom),
                        "edge_occupancy_rate": float(edge_ticks / denom),
                        "center_escape_count": int(center_escape),
                        "domain_randomization": dict(last_info.get("domain_randomization", {})) if last_info else {},
                        "vision_sensor_dr": dict(last_info.get("vision_sensor_dr", {})) if last_info else {},
                        "camera_frame_lag": last_info.get("camera_frame_lag") if last_info else None,
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
        return {"type": "eval", "worker": self.worker_id, "map_path": str(command["map_path"]), "results": results}


def worker_main(worker_id, conn, cfg):
    runtime = None
    try:
        runtime = WorkerRuntime(worker_id, cfg)
        current_world = runtime.client.get_world()
        conn.send({
            "type": "ready",
            "worker": worker_id,
            "port": cfg["port"],
            "current_map": str(current_world.get_map().name),
            "available_maps": runtime.available_maps(),
        })
        while True:
            command = conn.recv()
            kind = command.get("cmd")
            if kind == "close":
                break
            if kind == "collect":
                conn.send(runtime.collect(command))
            elif kind == "evaluate_policy":
                conn.send(runtime.evaluate_policy(command))
            else:
                raise RuntimeError("Unknown worker command: {}".format(kind))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            conn.send({"type": "error", "worker": worker_id, "error": repr(exc), "traceback": traceback.format_exc()})
        except Exception:
            pass
    finally:
        if runtime is not None:
            runtime.close()
        try:
            conn.close()
        except Exception:
            pass


# =============================================================================
# Evaluation helpers
# =============================================================================

def run_map_eval(parent_conns, master, args, maps, spawns, max_steps,
                 repeats=1, dynamics=False, vision=False, sensor=False,
                 global_step=0, seed_tag=0):
    snapshot = state_dict_to_numpy(master.old_policy)
    all_results = []
    maps = list(maps)
    worker_count = int(args.workers)

    for batch_start in range(0, len(maps), worker_count):
        batch = maps[batch_start:batch_start + worker_count]
        active = []
        for local_worker, map_path in enumerate(batch):
            worker_id = local_worker
            seed_base = int(global_step) + int(seed_tag) + batch_start * 1000 + local_worker * 100
            command = {
                "cmd": "evaluate_policy", "map_path": map_path,
                "safe_spawns": list(spawns), "max_steps": int(max_steps),
                "repeats": int(repeats), "policy_state": snapshot,
                "scenario_seed": 810000 + seed_base,
                "dr_seed": 910000 + seed_base,
                "dynamics_dr": bool(dynamics), "vision_dr": bool(vision),
                "sensor_dr": bool(sensor), "recovery": False, "recovery_prob": 0.0,
                "edge_ratio_threshold": float(args.canary_edge_ratio_threshold),
                "center_start_ratio": float(args.canary_center_start_ratio),
                "center_escape_ratio": float(args.canary_center_escape_ratio),
                "center_escape_window_ticks": int(args.canary_center_escape_window_ticks),
            }
            parent_conns[worker_id].send(command)
            active.append(worker_id)
        for worker_id in active:
            conn = parent_conns[worker_id]
            if not conn.poll(float(args.worker_timeout)):
                raise TimeoutError("Evaluation worker {} timeout".format(worker_id))
            msg = conn.recv()
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("traceback", msg.get("error")))
            if msg.get("type") != "eval":
                raise RuntimeError("Unexpected eval message: {}".format(msg))
            all_results.extend(msg["results"])
    return all_results


def judge_eval_case(item, args, mode):
    reasons = []
    reason = str(item.get("reason"))
    if reason in ("offroad", "collision", "stuck"):
        reasons.append("terminated={}".format(reason))

    if mode == "anchor":
        avg_lat = args.canary_avg_lateral_max_m
        max_lat = args.canary_max_lateral_max_m
        avg_head = args.canary_avg_heading_max_deg
        edge = args.canary_edge_rate_max
        steer075 = args.canary_steer075_rate_max
        min_progress = args.canary_min_progress_m
    elif mode == "mastery":
        avg_lat = args.mastery_avg_lateral_max_m
        max_lat = args.mastery_max_lateral_max_m
        avg_head = args.mastery_avg_heading_max_deg
        edge = args.mastery_edge_rate_max
        steer075 = args.mastery_steer075_rate_max
        min_progress = args.mastery_min_progress_m
    elif mode == "robust":
        avg_lat = args.robust_avg_lateral_max_m
        max_lat = args.robust_max_lateral_max_m
        avg_head = args.robust_avg_heading_max_deg
        edge = args.robust_edge_rate_max
        steer075 = args.robust_steer075_rate_max
        min_progress = args.robust_min_progress_m
    else:
        raise ValueError(mode)

    if float(item.get("avg_abs_lateral_m", 999.0)) > float(avg_lat):
        reasons.append("avg_lat")
    if float(item.get("max_abs_lateral_m", 999.0)) > float(max_lat):
        reasons.append("max_lat")
    if float(item.get("avg_abs_heading_deg", 999.0)) > float(avg_head):
        reasons.append("avg_heading")
    if float(item.get("edge_occupancy_rate", 1.0)) > float(edge):
        reasons.append("edge")
    if float(item.get("steer_gt_075_rate", 1.0)) > float(steer075):
        reasons.append("steer075")
    if float(item.get("progress_m", 0.0)) < float(min_progress):
        reasons.append("progress")
    if mode != "robust" and int(item.get("center_escape_count", 0)) > 0:
        reasons.append("center_escape")
    return len(reasons) == 0, reasons


def summarize_eval_cases(items):
    if not items:
        return {"cases": 0}
    return {
        "cases": len(items),
        "avg_abs_lateral_m": float(np.mean([float(x["avg_abs_lateral_m"]) for x in items])),
        "max_abs_lateral_m": float(max(float(x["max_abs_lateral_m"]) for x in items)),
        "avg_abs_heading_deg": float(np.mean([float(x["avg_abs_heading_deg"]) for x in items])),
        "max_abs_heading_deg": float(max(float(x["max_abs_heading_deg"]) for x in items)),
        "avg_abs_steer": float(np.mean([float(x["avg_abs_steer"]) for x in items])),
        "steer_gt_075_rate": float(np.mean([float(x["steer_gt_075_rate"]) for x in items])),
        "edge_occupancy_rate": float(np.mean([float(x["edge_occupancy_rate"]) for x in items])),
        "progress_m_mean": float(np.mean([float(x["progress_m"]) for x in items])),
        "offroad_count": int(sum(1 for x in items if str(x.get("reason")) == "offroad")),
    }


def judge_anchor(results, args):
    reasons = []
    expected = len(ANCHOR_MAPS) * len(args.safe_spawns)
    if len(results) != expected:
        reasons.append("anchor cases {} != expected {}".format(len(results), expected))

    # Fail loudly if an AUX map ever leaks into ANCHOR CLEAN CANARY.
    allowed = set(ANCHOR_MAPS)
    result_maps = [str(item.get("map_path")) for item in results]
    unexpected = sorted(set(result_maps) - allowed)
    if unexpected:
        reasons.append("ANCHOR MAP ROUTING ERROR unexpected={}".format(",".join(map_short(x) for x in unexpected)))
    counts = {m: 0 for m in ANCHOR_MAPS}
    for m in result_maps:
        if m in counts:
            counts[m] += 1
    expected_each = len(args.safe_spawns)
    for m in ANCHOR_MAPS:
        if counts[m] != expected_each:
            reasons.append("ANCHOR MAP ROUTING ERROR {} cases {} != {}".format(map_short(m), counts[m], expected_each))

    for item in results:
        ok, why = judge_eval_case(item, args, "anchor")
        if not ok:
            reasons.append("{} spawn{}: {}".format(map_short(item["map_path"]), item["spawn_number"], ",".join(why)))
    return len(reasons) == 0, reasons, summarize_eval_cases(results)


def judge_robust(results, args):
    grouped = defaultdict(list)
    for item in results:
        grouped[str(item["map_path"])].append(item)
    map_reports = {}
    reasons = []
    for map_path in ALL_MAPS:
        items = grouped.get(map_path, [])
        passed = sum(1 for x in items if judge_eval_case(x, args, "robust")[0])
        rate = float(passed) / float(len(items)) if items else 0.0
        map_reports[map_path] = {
            "cases": len(items), "passed": int(passed), "pass_rate": rate,
            "summary": summarize_eval_cases(items),
        }
        if rate < float(args.robust_map_pass_rate):
            reasons.append("{} robust pass {:.1f}%<{:.1f}%".format(
                map_short(map_path), 100*rate, 100*args.robust_map_pass_rate
            ))
    return len(reasons) == 0, reasons, map_reports


def print_eval_table(title, results, args, mode):
    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    grouped = defaultdict(list)
    for item in results:
        grouped[str(item["map_path"])].append(item)
    for map_path in ALL_MAPS:
        if map_path not in grouped:
            continue
        items = grouped[map_path]
        passed = sum(1 for x in items if judge_eval_case(x, args, mode)[0])
        s = summarize_eval_cases(items)
        print("  {:14s} | PASS={:2d}/{:2d} | offroad={:2d} | lat={:.4f}/{:.4f}m | head={:.2f}deg | |steer|={:.3f}".format(
            map_short(map_path), passed, len(items), int(s.get("offroad_count", 0)),
            float(s.get("avg_abs_lateral_m", 0.0)), float(s.get("max_abs_lateral_m", 0.0)),
            float(s.get("avg_abs_heading_deg", 0.0)), float(s.get("avg_abs_steer", 0.0))
        ))
    print("=" * 118)


# =============================================================================
# Main utility
# =============================================================================

def maybe_decay_action_std(agent, global_step, previous_bucket, args):
    bucket = int(global_step // int(args.action_std_decay_freq))
    if bucket <= previous_bucket:
        return previous_bucket
    for _ in range(bucket - previous_bucket):
        if agent.action_std <= float(args.action_std_min) + 1e-12:
            break
        agent.decay_action_std(float(args.action_std_decay), float(args.action_std_min))
    print("ACTION STD | step={} | std={:.4f}".format(global_step, agent.action_std))
    return bucket


def make_writer(args):
    if not bool(args.tensorboard) or SummaryWriter is None:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", "{}_{}".format(MODEL_NAME, stamp))
    writer = SummaryWriter(run_dir)
    writer.add_text("config", "\n".join("{} = {}".format(k, v) for k, v in sorted(vars(args).items())))
    print("TensorBoard:", run_dir)
    return writer


def worker_configs(args):
    sources = [(args.port0, args.seed0), (args.port1, args.seed1)]
    result = []
    lvl1 = recovery_level_params(1, args)
    for worker_id in range(int(args.workers)):
        port, seed = sources[worker_id]
        result.append({
            "host": args.host, "port": int(port), "seed": int(seed),
            "carla_timeout": float(args.carla_timeout),
            "max_episode_seconds": float(args.max_episode_seconds),
            "encoder_device": args.encoder_device,
            "worker_ppo_device": args.worker_ppo_device,
            "worker_torch_threads": int(args.worker_torch_threads),
            "action_std_init": float(args.action_std_init),
            "recovery_steer_min": lvl1["steer_min"],
            "recovery_steer_max": lvl1["steer_max"],
            "recovery_duration_min_s": lvl1["duration_min_s"],
            "recovery_duration_max_s": lvl1["duration_max_s"],
            "recovery_min_episode_step": int(args.recovery_min_episode_step),
            "recovery_stable_ticks": int(args.recovery_stable_ticks),
            "recovery_stable_lateral_m": float(args.recovery_stable_lateral_m),
            "recovery_stable_heading_deg": float(args.recovery_stable_heading_deg),
            "recovery_stable_speed_mps": float(args.recovery_stable_speed_mps),
            "recovery_cooldown_ticks": int(args.recovery_cooldown_ticks),
            "recovery_max_events": int(args.recovery_max_events),
            "latent_audit": bool(args.latent_audit),
            "latent_sensitivity_every_steps": int(args.latent_sensitivity_every_steps),
            "latent_sensitivity_epsilon_l2": float(args.latent_sensitivity_epsilon_l2),
            "latent_sensitivity_alert": float(args.latent_sensitivity_alert),
        })
    return result


def print_mastery(mastery, weights, args):
    print("  MAP MASTERY / NEXT WEIGHT")
    for map_path in ALL_MAPS:
        rec = mastery.maps[map_path]
        label = "MASTERED" if map_is_mastered(mastery, map_path, args) else str(rec["status"])
        robust_label = str(rec.get("robust_status", STATUS_UNKNOWN))
        print("    {:14s} {:8s} streak={} clean={:5.1f}% robust={:7s} weight={:5.1f}%".format(
            map_short(map_path), label, int(rec["pass_streak"]),
            100.0 * float(rec.get("pass_rate", 0.0)), robust_label,
            100.0 * float(weights.get(map_path, 0.0))
        ))


def print_recovery_audit(audit):
    data = audit.to_dict()
    total = data["injected_pos"] + data["injected_neg"]
    succ = data["success_pos"] + data["success_neg"]
    print("  RECOVERY AUDIT | injected +={} -={} | success +={} -={} | overall={:.1f}%".format(
        data["injected_pos"], data["injected_neg"], data["success_pos"], data["success_neg"],
        100.0 * succ / float(total) if total else 0.0
    ))
    for state in ("stable", "mild", "moderate", "strong"):
        n = int(data["state_counts"][state])
        if n > 0:
            print("    {:8s} n={:7d} |cmd-target|={:.3f} |v-target|={:.3f}".format(
                state.upper(), n,
                float(data["state_cmd_error_sum"][state]) / n,
                float(data["state_speed_error_sum"][state]) / n,
            ))


def print_latent_audit(audit, args):
    s = audit.summary()
    print(
        "  LATENT95 AUDIT | n={} | norm mean/max={:.3f}/{:.3f} | "
        "delta mean/max={:.3f}/{:.3f} | steer-sens mean/max={:.3f}/{:.3f} | alerts={}".format(
            int(audit.count), float(s["norm_mean"]), float(s["norm_max"]),
            float(s["delta_mean"]), float(s["delta_max"]),
            float(s["sensitivity_mean"]), float(s["sensitivity_max"]),
            int(s["sensitivity_alert_count"]),
        )
    )
    if int(s["failure_count"]) > 0:
        print(
            "    failures={} | latent norm mean={:.3f} | delta mean={:.3f}".format(
                int(s["failure_count"]), float(s["failure_norm_mean"]),
                float(s["failure_delta_mean"]),
            )
        )
    if float(s["sensitivity_max"]) >= float(args.latent_sensitivity_alert):
        print(
            "    LATENT SENSITIVITY WARNING | max {:.3f} >= alert {:.3f} | diagnostic only".format(
                float(s["sensitivity_max"]), float(args.latent_sensitivity_alert)
            )
        )


def print_dr_audit(exposure, args):
    passed, reasons, s = exposure.gate(args)
    print("  DR EXPOSURE | full={}/{} ({:.1f}%) | episode_samples={} | gate={}".format(
        s["full_steps"], s["total_steps"], 100.0*s["full_ratio"], s["dr_episode_samples"],
        "PASS" if passed else "NOT_READY"
    ))
    if not passed:
        for reason in reasons[:4]:
            print("    - " + reason)
    return passed


def advance_recovery_level_if_ready(recovery_level, recovery_audit, args):
    if int(recovery_level) >= int(args.recovery_level_max):
        return int(recovery_level), False
    passed, _, _ = recovery_audit.gate(args, level_gate=True)
    if not passed:
        return int(recovery_level), False
    return int(recovery_level) + 1, True


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    validate_args(args)
    mp.freeze_support()
    torch.set_num_threads(int(args.learner_torch_threads))
    seed_everything(int(SEED))

    master = PPOAgent(town=MODEL_NAME, action_std_init=float(args.action_std_init), device=args.learner_device)
    state = prepare_fresh_or_resume(master, args)
    global_step = state["global_step"]
    episode = state["episode"]
    stage = state["stage"]
    stage_start_step = state["stage_start_step"]
    consecutive_anchor_passes = state["consecutive_anchor_passes"]
    loaded_checkpoint = state["loaded_checkpoint"]
    current_checkpoint = state["current_checkpoint"]
    mastery = state["mastery"]
    exposure = state["exposure"]
    failure_clusters = state["failure_clusters"]
    recovery_audit = state["recovery_audit"]
    latent_audit = state["latent_audit"]
    recovery_level = state["recovery_level"]
    stage3_fail_streak = state["stage3_fail_streak"]
    repair_until_step = state["repair_until_step"]

    # Explicit operator override for this run: leave Stage2 immediately and
    # persist Stage3 before any rollout starts.  This intentionally bypasses
    # only the Stage2 clean-mastery gate; all Stage3 robust/DR verification,
    # adaptive map weighting, recovery, canaries and champion logic stay active.
    if bool(args.force_stage3_now):
        if not bool(args.resume):
            raise RuntimeError("--force-stage3-now requires --resume true")
        if int(stage) < 2:
            raise RuntimeError("--force-stage3-now is only valid from Stage2/Stage3; current stage={}".format(
                STAGE_NAMES[int(stage)]
            ))
        if int(stage) == 2:
            old_stage = int(stage)
            stage = 3
            stage_start_step = int(global_step)
            consecutive_anchor_passes = 0
            stage3_fail_streak = 0
            repair_until_step = 0
            recovery_audit.reset()
            print("FORCE CURRICULUM ADVANCE | {} -> {} | step={} | operator_override=true".format(
                STAGE_NAMES[old_stage], STAGE_NAMES[int(stage)], int(global_step)
            ))
            forced_state = write_state(
                master, global_step, episode, stage, stage_start_step,
                consecutive_anchor_passes, current_checkpoint, mastery,
                exposure, failure_clusters, recovery_audit, latent_audit, recovery_level,
                stage3_fail_streak, repair_until_step, args,
            )
            print("FORCE STAGE3 STATE SAVED | {}".format(forced_state))
            print("FULL SIM2REAL DR REQUESTED | dynamics={} vision={} sensor={}".format(
                bool(args.dynamics_dr), bool(args.vision_dr), bool(args.sensor_dr)
            ))
        else:
            print("FORCE STAGE3 | already in {} at step={}; continuing without changing state".format(
                STAGE_NAMES[int(stage)], int(global_step)
            ))

    target_step = int(args.total_timesteps)
    writer = make_writer(args)
    cfgs = worker_configs(args)
    ctx = mp.get_context("spawn")
    parent_conns = []
    processes = []
    run_start = time.time()

    try:
        for worker_id in range(int(args.workers)):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            process = ctx.Process(
                target=worker_main, args=(worker_id, child_conn, cfgs[worker_id]),
                name="CARLA-V5-VNEXT-W{}".format(worker_id)
            )
            process.daemon = False
            process.start()
            child_conn.close()
            parent_conns.append(parent_conn)
            processes.append(process)

        ready = []
        for worker_id, conn in enumerate(parent_conns):
            if not conn.poll(float(args.worker_timeout)):
                raise TimeoutError("Worker {} init timeout".format(worker_id))
            msg = conn.recv()
            if msg.get("type") == "error":
                raise RuntimeError(msg.get("traceback", msg.get("error")))
            ready.append(msg)

        # Fail fast before spending hundreds of thousands of transitions.
        # A configured path may be auto-resolved only when CARLA exposes one
        # unambiguous equivalent map asset. Required maps are never skipped.
        reference_maps = ready[0].get("available_maps", []) if ready else []
        print("\nCARLA MAP PREFLIGHT | configured={} | available={}".format(
            len(ALL_MAPS), len(reference_maps)
        ))
        unresolved = []
        for requested_map in ALL_MAPS:
            resolved_map, hints = resolve_map_path_from_available(requested_map, reference_maps)
            if resolved_map is None:
                unresolved.append((requested_map, hints))
                print("  MISSING  | {}".format(requested_map))
                for hint in hints[:6]:
                    print("           ? {}".format(hint))
            elif str(resolved_map).replace("\\", "/").lower() == str(requested_map).replace("\\", "/").lower():
                print("  OK       | {}".format(requested_map))
            else:
                print("  RESOLVED | {} -> {}".format(requested_map, resolved_map))
        if unresolved:
            lines = ["CARLA MAP PREFLIGHT FAILED. Correct these configured paths before training:"]
            for requested_map, hints in unresolved:
                lines.append("  - {}".format(requested_map))
                if hints:
                    lines.append("    similar: {}".format(", ".join(hints[:6])))
            raise RuntimeError("\n".join(lines))
        print("CARLA MAP PREFLIGHT | PASS | all required maps available/resolvable\n")
        print("VNEXT MAP ROLES | anchors={} | aux={} | stress={}".format(
            [map_short(x) for x in ANCHOR_MAPS],
            [map_short(x) for x in AUX_OVAL_MAPS],
            map_short(LEFT_ONLY_STRESS_MAP),
        ))
        print("VNEXT MAP UNIQUE CHECK | {} unique maps".format(len(ALL_MAPS)))

        print("=" * 118)
        print("PPO V5 VNEXT MAIN TRAINER | FRESH / GLOBAL ADAPTIVE MULTIMAP")
        print("=" * 118)
        print("model             :", MODEL_NAME)
        print("loaded            :", loaded_checkpoint if loaded_checkpoint else "RANDOM INIT")
        print("global/target     : {} / {}".format(global_step, target_step))
        print("stage             : {} ({})".format(stage, STAGE_NAMES[stage]))
        print("spawns            : {} CLEAN ONLY".format(list(args.safe_spawns)))
        print("maps              : {} total | anchors={} aux={} stress=1".format(len(ALL_MAPS), len(ANCHOR_MAPS), len(AUX_OVAL_MAPS)))
        print("scheduler         : GLOBAL ADAPTIVE + INTERLEAVED 100K MAP BLOCKS")
        print("map switch        : every {:,} transitions | fixed map inside each block".format(int(args.map_block_steps)))
        print("map probability   : anchor floor={:.0f}% each | other floor={:.0f}% each | max={:.0f}% | stress max={:.0f}%".format(
            100*args.map_floor_anchor, 100*args.map_floor_other,
            100*args.map_max_weight, 100*args.stress_max_weight
        ))
        print("map mastery gate  : every {} steps | {} ticks | spawns={} | streak={}".format(
            args.map_validation_every_steps, args.mastery_max_steps, args.mastery_spawns, args.mastery_pass_count
        ))
        print("robust DR gate    : every {} steps | spawns={} x repeats={}".format(
            args.robust_validation_every_steps, args.robust_spawns, args.robust_repeats
        ))
        print("category champions: CLEAN / MULTIMAP / RECOVERY / SIM2REAL are separate")
        print("=" * 118)

        if args.smoke_test:
            target_step = global_step + int(args.smoke_steps_per_worker) * int(args.workers)

        # Optional initial full clean map census when fine-tuning directly in Stage2/3.
        if (
            not args.smoke_test and bool(args.initial_map_validation)
            and stage >= 2 and all(rec["last_eval_step"] < 0 for rec in mastery.maps.values())
        ):
            print("\nINITIAL MAP MASTERY CENSUS | exact loaded policy, DR OFF, recovery OFF")
            initial_results = run_map_eval(
                parent_conns, master, args, ALL_MAPS, args.mastery_spawns,
                args.mastery_max_steps, repeats=1, dynamics=False, vision=False,
                sensor=False, global_step=global_step, seed_tag=101,
            )
            mastery.update(initial_results, args, global_step)
            print_eval_table("INITIAL CLEAN MAP MASTERY", initial_results, args, "mastery")

        next_canary_step = ((global_step // int(args.canary_every_steps)) + 1) * int(args.canary_every_steps)
        next_map_validation_step = ((global_step // int(args.map_validation_every_steps)) + 1) * int(args.map_validation_every_steps)
        next_robust_validation_step = ((global_step // int(args.robust_validation_every_steps)) + 1) * int(args.robust_validation_every_steps)
        decay_bucket = int(global_step // int(args.action_std_decay_freq))
        update_index = 0

        # Cache one assignment for the entire map block.  This is what makes
        # "switch every 100k" strict instead of allowing a validation/failure
        # update to silently change maps halfway through a block.
        scheduler_cache_key = None
        scheduler_block_weights = None
        scheduler_block_maps = None

        while global_step < target_step:
            update_index += 1
            requested_total = min(int(args.rollout_total), int(target_step - global_step))
            per_worker = split_steps(requested_total, int(args.workers))

            repair_mode = bool(stage == 3 and global_step < int(repair_until_step))
            stage_cfg = stage_config(stage, args, repair_mode=repair_mode, recovery_level=recovery_level)
            cluster_maps = failure_clusters.alert_maps(args)
            fresh_map_weights = adaptive_map_weights(mastery, args, cluster_maps)
            stage_block_index = int(max(0, global_step - stage_start_step) // int(args.map_block_steps))
            scheduler_key = (int(stage), int(stage_start_step), int(stage_block_index))

            if scheduler_key != scheduler_cache_key:
                scheduler_cache_key = scheduler_key
                scheduler_block_weights = dict(fresh_map_weights)
                scheduler_block_maps = assigned_maps_for_block(
                    stage, stage_block_index, int(args.workers), scheduler_block_weights, args
                )

                block_start = int(stage_start_step) + int(stage_block_index) * int(args.map_block_steps)
                block_end = block_start + int(args.map_block_steps)
                print("\nMAP SCHEDULER BLOCK | stage={} | block={} | steps=[{:,},{:,}) | switch_every={:,}".format(
                    STAGE_NAMES[int(stage)], int(stage_block_index), block_start, block_end, int(args.map_block_steps)
                ))
                print("  ASSIGNMENTS | " + " | ".join(
                    "W{}={}".format(i, map_short(m)) for i, m in enumerate(scheduler_block_maps)
                ))
                if int(stage) >= 2:
                    ordered = sorted(ALL_MAPS, key=lambda m: scheduler_block_weights[m], reverse=True)
                    print("  ADAPTIVE %  | " + " | ".join(
                        "{}={:.1f}%".format(map_short(m), 100.0 * scheduler_block_weights[m])
                        for m in ordered
                    ))
                    print("  RULE        | FAIL/WEAK ↑ | PASS ↓ | MASTERED ↓↓ | floors prevent forgetting")

            map_weights = scheduler_block_weights
            policy_snapshot = state_dict_to_numpy(master.old_policy)
            rec_params = recovery_level_params(recovery_level, args)

            active = []
            collect_start = time.time()
            for worker_id, steps in enumerate(per_worker):
                if steps <= 0:
                    continue
                map_path = scheduler_block_maps[worker_id]
                scenario_seed = int(cfgs[worker_id]["seed"]) + stage*100000 + stage_block_index*1000
                dr_seed = scenario_seed + 60000
                parent_conns[worker_id].send({
                    "cmd": "collect", "steps": int(steps), "policy_state": policy_snapshot,
                    "action_std": float(master.action_std), "map_path": map_path,
                    "safe_spawns": list(args.safe_spawns), "desired_speed": float(args.desired_speed),
                    "recovery": bool(stage_cfg["recovery"]), "recovery_prob": float(stage_cfg["recovery_prob"]),
                    "recovery_steer_min": rec_params["steer_min"], "recovery_steer_max": rec_params["steer_max"],
                    "recovery_duration_min_s": rec_params["duration_min_s"], "recovery_duration_max_s": rec_params["duration_max_s"],
                    "dynamics_dr": bool(stage_cfg["dynamics_dr"]), "vision_dr": bool(stage_cfg["vision_dr"]),
                    "sensor_dr": bool(stage_cfg["sensor_dr"]),
                    "scenario_seed": int(scenario_seed), "dr_seed": int(dr_seed),
                })
                active.append(worker_id)

            messages = []
            for worker_id in active:
                conn = parent_conns[worker_id]
                if not conn.poll(float(args.worker_timeout)):
                    raise TimeoutError("Worker {} rollout timeout".format(worker_id))
                msg = conn.recv()
                if msg.get("type") == "error":
                    raise RuntimeError(msg.get("traceback", msg.get("error")))
                if msg.get("type") != "rollout":
                    raise RuntimeError("Unexpected worker message")
                messages.append(msg)

            collect_wall = time.time() - collect_start
            if sum(int(m["samples"]) for m in messages) != requested_total:
                raise RuntimeError("Rollout sample mismatch")

            diagnostics = aggregate_diagnostics(messages)
            episode_metrics = aggregate_episode_metrics(messages)
            if args.smoke_test:
                print("\nSMOKE PASS | maps={}".format([map_short(m["map_path"]) for m in messages]))
                print_state_diagnostics(diagnostics)
                break

            master.memory.clear()
            merged = 0
            for msg in sorted(messages, key=lambda x: int(x["worker"])):
                merged += append_rollout_to_master(master, msg["rollout"])
            if merged != requested_total:
                raise RuntimeError("Merged rollout mismatch")
            ppo_metrics = master.learn(last_obs=None)

            # Audits refer to the transitions just used for this update.
            for msg in messages:
                exposure.add_rollout(
                    msg["map_path"], msg["samples"], msg["dynamics_dr"],
                    msg["vision_dr"], msg["sensor_dr"]
                )
                for sample in msg.get("dr_episode_samples", []):
                    exposure.add_episode_sample(sample)
                recovery_audit.update_export(msg.get("recovery_audit", {}))
                latent_audit.update_export(msg.get("latent_audit", {}))
                for event in msg.get("failure_events", []):
                    failure_clusters.add(event)

            global_step += requested_total
            episode += int(episode_metrics.get("episodes", 0))
            decay_bucket = maybe_decay_action_std(master, global_step, decay_bucket, args)

            print("\nUPDATE {:05d} | step={}/{} | stage={}{} | block={} | maps={} | samples={} collect={:.1f}s SPS={:.1f} std={:.4f}".format(
                update_index, global_step, target_step, STAGE_NAMES[stage],
                "[REPAIR]" if repair_mode else "", stage_block_index,
                [map_short(m["map_path"]) for m in messages], requested_total,
                collect_wall, requested_total / max(collect_wall, 1e-9), master.action_std
            ))
            print("  PPO | loss={:+.5f} policy={:+.5f} value={:.5f} entropy={:.5f} return={:+.5f}".format(
                float(ppo_metrics.get("loss", 0.0)), float(ppo_metrics.get("policy_loss", 0.0)),
                float(ppo_metrics.get("value_loss", 0.0)), float(ppo_metrics.get("entropy", 0.0)),
                float(ppo_metrics.get("mean_return", 0.0))
            ))
            print_state_diagnostics(diagnostics)

            if writer is not None:
                for key, value in ppo_metrics.items():
                    writer.add_scalar("ppo/{}".format(key), float(value), global_step)
                writer.add_scalar("curriculum/stage", float(stage), global_step)
                writer.add_scalar("curriculum/recovery_level", float(recovery_level), global_step)
                writer.add_scalar("dr/full_steps", float(exposure.full_steps), global_step)
                writer.add_scalar("dr/full_ratio", float(exposure.full_steps)/max(exposure.total_steps,1), global_step)
                write_tensorboard_diagnostics(writer, diagnostics, global_step)

            # Keep Stage1 short: learn only basic delayed-recovery Level1 on
            # anchors. Stronger recovery levels are learned after multimap starts
            # so hard auxiliary maps are not starved by prolonged anchor-only work.
            if stage >= 2:
                new_level, advanced_level = advance_recovery_level_if_ready(
                    recovery_level, recovery_audit, args
                )
                if advanced_level:
                    print("RECOVERY LEVEL ADVANCE | {} -> {} | audit reset for new level".format(recovery_level, new_level))
                    recovery_level = new_level
                    recovery_audit.reset()

            gate_due = bool(global_step >= next_canary_step or global_step >= target_step)
            if not gate_due:
                continue

            evaluated_stage = int(stage)
            anchor_results = run_map_eval(
                parent_conns, master, args, ANCHOR_MAPS, args.safe_spawns,
                args.canary_max_steps, repeats=1, dynamics=False, vision=False,
                sensor=False, global_step=global_step, seed_tag=201,
            )
            anchor_pass, anchor_reasons, anchor_summary = judge_anchor(anchor_results, args)
            print_eval_table("ANCHOR CLEAN CANARY | {}".format("PASS" if anchor_pass else "FAIL"), anchor_results, args, "anchor")
            if not anchor_pass:
                for reason in anchor_reasons[:12]:
                    print("  FAIL | " + reason)

            map_validation_due = bool(
                stage >= 2 and (global_step >= next_map_validation_step or global_step >= target_step)
            )
            robust_due = bool(
                stage == 3 and not repair_mode
                and (global_step >= next_robust_validation_step or global_step >= target_step)
            )

            verification = {
                "clean_anchor": bool(anchor_pass),
                "multimap": False,
                "recovery": False,
                "sim2real": False,
                "anchor_summary": anchor_summary,
            }

            if anchor_pass:
                consecutive_anchor_passes += 1
                stage3_fail_streak = 0

                # Run expensive map mastery only on a clean-anchor candidate.
                if map_validation_due:
                    map_results = run_map_eval(
                        parent_conns, master, args, ALL_MAPS, args.mastery_spawns,
                        args.mastery_max_steps, repeats=1, dynamics=False, vision=False,
                        sensor=False, global_step=global_step, seed_tag=301,
                    )
                    mastery.update(map_results, args, global_step)
                    print_eval_table("FULL CLEAN MAP MASTERY", map_results, args, "mastery")
                    while next_map_validation_step <= global_step:
                        next_map_validation_step += int(args.map_validation_every_steps)

                multimap_pass, missing_maps = mastery.required_maps_mastered(args)
                recovery_pass, recovery_reasons, recovery_summary = recovery_audit.gate(args, level_gate=False)
                verification["multimap"] = bool(multimap_pass)
                verification["recovery"] = bool(recovery_pass)
                verification["missing_maps"] = [map_short(x) for x in missing_maps]
                verification["recovery_summary"] = recovery_summary
                if stage == 2 and not multimap_pass:
                    print("  MULTIMAP GATE NOT READY | missing={}".format(
                        ",".join(map_short(x) for x in missing_maps) if missing_maps else "<unknown>"
                    ))

                # Make Stage1/Stage2 stalls explicit in the console.  This is
                # diagnostic only; it does not change the curriculum gate.
                if stage in (1, 2) and not recovery_pass:
                    print("  RECOVERY GATE NOT READY | " + " | ".join(recovery_reasons))

                robust_pass = False
                robust_reasons = ["not evaluated"]
                robust_reports = {}
                if robust_due:
                    robust_results = run_map_eval(
                        parent_conns, master, args, ALL_MAPS, args.robust_spawns,
                        args.robust_max_steps, repeats=int(args.robust_repeats),
                        dynamics=bool(args.dynamics_dr), vision=bool(args.vision_dr),
                        sensor=bool(args.sensor_dr), global_step=global_step, seed_tag=401,
                    )
                    robust_pass, robust_reasons, robust_reports = judge_robust(robust_results, args)
                    mastery.update_robust(robust_reports, args, global_step)
                    print_eval_table("ROBUST FULL-DR MAP VALIDATION | {}".format("PASS" if robust_pass else "FAIL"), robust_results, args, "robust")
                    while next_robust_validation_step <= global_step:
                        next_robust_validation_step += int(args.robust_validation_every_steps)

                dr_pass, dr_reasons, dr_summary = exposure.gate(args)
                verification["robust"] = bool(robust_pass)
                verification["robust_reasons"] = list(robust_reasons)
                verification["dr_coverage"] = bool(dr_pass)
                verification["dr_summary"] = dr_summary
                verification["sim2real"] = bool(multimap_pass and recovery_pass and robust_pass and dr_pass)

                # Stage advance: minimum step + appropriate verified gates.
                stage_steps = int(global_step - stage_start_step)
                old_stage = int(stage)
                if stage == 0:
                    if stage_steps >= stage_min_steps(stage, args) and consecutive_anchor_passes >= int(args.stage_pass_count):
                        stage = 1
                elif stage == 1:
                    if (
                        stage_steps >= stage_min_steps(stage, args)
                        and consecutive_anchor_passes >= int(args.stage_pass_count)
                        and recovery_pass
                    ):
                        stage = 2
                elif stage == 2:
                    if (
                        stage_steps >= stage_min_steps(stage, args)
                        and consecutive_anchor_passes >= int(args.stage_pass_count)
                        and multimap_pass and recovery_pass
                    ):
                        stage = 3
                if stage != old_stage:
                    print("CURRICULUM ADVANCE | {} -> {} | step={}".format(STAGE_NAMES[old_stage], STAGE_NAMES[stage], global_step))
                    stage_start_step = int(global_step)
                    consecutive_anchor_passes = 0
                    scheduler_cache_key = None
                    scheduler_block_weights = None
                    scheduler_block_maps = None
                    # Keep mastery/DR cumulative; reset stage-local recovery evidence.
                    recovery_audit.reset()

                    # The first Stage2 training block must already know which
                    # auxiliary maps are hard. Do a clean census immediately
                    # instead of spending another fixed interval uniformly.
                    if stage == 2:
                        print("STAGE2 ENTRY MAP CENSUS | all maps x mastery spawns | DR OFF")
                        entry_results = run_map_eval(
                            parent_conns, master, args, ALL_MAPS, args.mastery_spawns,
                            args.mastery_max_steps, repeats=1, dynamics=False,
                            vision=False, sensor=False, global_step=global_step,
                            seed_tag=302,
                        )
                        mastery.update(entry_results, args, global_step)
                        print_eval_table("STAGE2 ENTRY CLEAN MAP MASTERY", entry_results, args, "mastery")
                        while next_map_validation_step <= global_step:
                            next_map_validation_step += int(args.map_validation_every_steps)

                # A numbered checkpoint means CLEAN ANCHOR VERIFIED only. Category
                # champions carry the stronger verification meaning.
                passed_path = save_numbered_checkpoint(master, global_step, evaluated_stage, verification)
                current_checkpoint = passed_path
                save_category_champion("clean", master, global_step, evaluated_stage, passed_path, verification)
                if recovery_pass:
                    save_category_champion("recovery", master, global_step, evaluated_stage, passed_path, verification)
                if multimap_pass:
                    save_category_champion("multimap", master, global_step, evaluated_stage, passed_path, verification)
                if verification["sim2real"]:
                    save_category_champion("sim2real", master, global_step, evaluated_stage, passed_path, verification)
                    print("SIM2REAL VERIFIED CHAMPION | {}".format(champion_path("sim2real")))

                state_file = write_state(
                    master, global_step, episode, stage, stage_start_step,
                    consecutive_anchor_passes, current_checkpoint, mastery,
                    exposure, failure_clusters, recovery_audit, latent_audit, recovery_level,
                    stage3_fail_streak, repair_until_step, args,
                )
                print("STATE SAVED | {}".format(state_file))

            else:
                consecutive_anchor_passes = 0
                rejected = save_rejected(master, global_step, stage, anchor_reasons, args.keep_rejected)
                print("ANCHOR CANDIDATE REJECTED | {}".format(rejected))
                kind = highest_available_champion(stage)
                if kind is not None:
                    restored = load_category_champion(kind, master)
                    current_checkpoint = str(restored.get("checkpoint_path") or current_checkpoint)
                    # Rollback must not resurrect an older exploration std.
                    champion_bucket = int(
                        int(restored.get("global_step", 0))
                        // int(args.action_std_decay_freq)
                    )
                    decay_bucket = maybe_decay_action_std(
                        master, global_step, champion_bucket, args
                    )
                    print("ROLLBACK {} CHAMPION | step={} policy={}".format(kind.upper(), restored.get("global_step"), current_checkpoint))
                else:
                    print("NO CATEGORY CHAMPION YET | continue current stage UNVERIFIED")

                if stage == 3:
                    stage3_fail_streak += 1
                    if stage3_fail_streak >= int(args.stage3_anchor_fail_patience):
                        repair_until_step = int(global_step) + int(args.stage3_repair_steps)
                        stage3_fail_streak = 0
                        print("STAGE3 REPAIR MODE | DR OFF for next {} steps, stage remains Stage3".format(args.stage3_repair_steps))

                write_state(
                    master, global_step, episode, stage, stage_start_step,
                    consecutive_anchor_passes, current_checkpoint, mastery,
                    exposure, failure_clusters, recovery_audit, latent_audit, recovery_level,
                    stage3_fail_streak, repair_until_step, args,
                )

            # Periodic supervision report.
            map_weights = adaptive_map_weights(mastery, args, failure_clusters.alert_maps(args))
            print_mastery(mastery, map_weights, args)
            print_recovery_audit(recovery_audit)
            if bool(args.latent_audit):
                print_latent_audit(latent_audit, args)
            print_dr_audit(exposure, args)
            top_clusters = failure_clusters.top_clusters(5)
            if top_clusters:
                print("  FAILURE CLUSTERS")
                for rec in top_clusters:
                    print("    {:14s} spawn{} count={} mean=({:.2f},{:.2f}) progress={:.2f}m".format(
                        map_short(rec["map_path"]), rec["spawn_number"], rec["count"],
                        rec["mean_x"], rec["mean_y"], rec["mean_progress_m"]
                    ))

            while next_canary_step <= global_step:
                next_canary_step += int(args.canary_every_steps)

        if not args.smoke_test:
            # Final status is explicit; completing a timestep budget is not the same
            # as being Sim2Real verified.
            multimap_pass, missing_maps = mastery.required_maps_mastered(args)
            recovery_pass, _, _ = recovery_audit.gate(args, level_gate=False)
            dr_pass, _, _ = exposure.gate(args)
            sim2real_exists = os.path.isfile(champion_path("sim2real"))
            print("\n" + "=" * 118)
            print("TRAINING COMPLETE | step={} | episode={} | stage={} | elapsed={:.1f} min".format(
                global_step, episode, STAGE_NAMES[stage], (time.time()-run_start)/60.0
            ))
            print("FINAL VERIFICATION")
            print("  CLEAN CHAMPION   : {}".format("YES" if os.path.isfile(champion_path("clean")) else "NO"))
            print("  MULTIMAP MASTERED: {}{}".format("YES" if multimap_pass else "NO", "" if multimap_pass else " | missing=" + ",".join(map_short(x) for x in missing_maps)))
            print("  RECOVERY GATE    : {}".format("PASS" if recovery_pass else "NOT_READY"))
            print("  DR COVERAGE GATE : {}".format("PASS" if dr_pass else "NOT_READY"))
            print("  SIM2REAL CHAMPION: {}".format("YES" if sim2real_exists else "NO"))
            print("  DEPLOY STATUS    : {}".format("SIM2REAL VERIFIED" if sim2real_exists else "NOT VERIFIED"))
            print("=" * 118)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        if not args.smoke_test:
            try:
                emergency = save_emergency(master, global_step, stage, "keyboard_interrupt", args.keep_emergency)
                print("EMERGENCY SNAPSHOT | UNVERIFIED | {}".format(emergency))
            except Exception:
                logging.exception("Could not save emergency snapshot")
    except Exception:
        logging.exception("VNext trainer failed")
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
        print("VNext cleanup complete.")


if __name__ == "__main__":
    main()
