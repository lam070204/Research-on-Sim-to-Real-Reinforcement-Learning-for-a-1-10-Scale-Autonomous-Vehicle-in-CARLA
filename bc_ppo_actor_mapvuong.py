# -*- coding: utf-8 -*-
"""
MAPVUONG BEHAVIOR-CLONE THE REAL PPO ACTOR

Purpose
-------
The oracle student probe proved that obs100 can reproduce the successful
pure-pursuit steering policy and run closed-loop 4/4 on mapvuong.

This tool now trains the *actual PPO actor network* offline on the same
oracle demonstrations, while preserving the source PPO speed behavior by
distillation.

Important
---------
- Source PPO checkpoint is NEVER modified.
- Existing PPO training_state JSON is NEVER modified.
- New checkpoint is saved under a NEW model namespace.
- Waypoints/oracle are used only as offline labels.
- PPO observation/action architecture stays unchanged.
"""

from __future__ import print_function

import argparse
import copy
import glob
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from networks.on_policy.ppo.ppo_agent_v5 import PPOAgent
from simulation.carla_connection_v5 import carla
from simulation.carla_environment_rgb_v5_singlemap_stable import (
    CarlaEnvironmentRGBV5SingleMapStable,
)


MAP_PATH = "/Game/mapvuong/mapvuong"
SAFE_SPAWNS = [1, 2, 3, 4]
OBS_DIM = 100


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def latest_dataset(project_root):
    pattern = os.path.join(
        project_root,
        "oracle_student_results",
        "student_probe_*",
        "oracle_obs100_dataset.npz",
    )
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            "No oracle_obs100_dataset.npz found under: {}".format(
                pattern
            )
        )
    candidates.sort(
        key=lambda p: os.path.getmtime(p)
    )
    return candidates[-1]


def get_actor(agent):
    policy = getattr(agent, "policy", None)
    if policy is None:
        raise RuntimeError(
            "PPOAgent has no .policy attribute."
        )

    actor = getattr(policy, "actor", None)
    if actor is None:
        raise RuntimeError(
            "agent.policy has no .actor attribute."
        )
    return actor


def get_old_actor(agent):
    old_policy = getattr(
        agent,
        "old_policy",
        None,
    )
    if old_policy is None:
        raise RuntimeError(
            "PPOAgent has no .old_policy attribute."
        )

    actor = getattr(
        old_policy,
        "actor",
        None,
    )
    if actor is None:
        raise RuntimeError(
            "agent.old_policy has no .actor attribute."
        )
    return actor


def transform_actor_raw(raw):
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise RuntimeError(
            "Expected PPO actor raw output [N,2], got {}".format(
                tuple(raw.shape)
            )
        )

    steer = torch.tanh(
        raw[:, 0]
    )
    speed_unit = 0.5 * (
        torch.tanh(
            raw[:, 1]
        )
        + 1.0
    )

    return steer, speed_unit


def split_mask(spawns, steps):
    block_id = steps // 100
    val_mask = (
        (block_id % 5) == 4
    )
    train_mask = ~val_mask

    if not np.any(val_mask):
        raise RuntimeError(
            "Validation split is empty."
        )
    if not np.any(train_mask):
        raise RuntimeError(
            "Training split is empty."
        )
    return train_mask, val_mask


def regression_metrics(pred, target):
    pred = np.asarray(
        pred,
        dtype=np.float32,
    ).reshape(-1)
    target = np.asarray(
        target,
        dtype=np.float32,
    ).reshape(-1)

    err = np.abs(
        pred - target
    )
    strong = np.abs(
        target
    ) >= 0.15

    corr = 0.0
    if pred.size >= 2:
        if (
            float(np.std(pred)) > 1e-8
            and float(np.std(target)) > 1e-8
        ):
            corr = float(
                np.corrcoef(
                    pred,
                    target,
                )[0, 1]
            )

    return {
        "mae": float(
            np.mean(err)
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    (pred - target) ** 2
                )
            )
        ),
        "strong_count": int(
            np.sum(strong)
        ),
        "strong_mae": (
            float(
                np.mean(
                    err[strong]
                )
            )
            if np.any(strong)
            else 0.0
        ),
        "strong_sign_acc": (
            float(
                np.mean(
                    np.sign(
                        pred[strong]
                    )
                    == np.sign(
                        target[strong]
                    )
                )
            )
            if np.any(strong)
            else 0.0
        ),
        "corr": corr,
    }


def evaluate_actor_arrays(
    actor,
    X,
    y_steer,
    y_speed,
    device,
    batch_size=1024,
):
    actor.eval()
    pred_steer = []
    pred_speed = []

    with torch.no_grad():
        for start in range(
            0,
            int(X.shape[0]),
            int(batch_size),
        ):
            xb = torch.from_numpy(
                X[
                    start:
                    start + int(batch_size)
                ]
            ).to(device)

            raw = actor(xb)
            s, v = transform_actor_raw(
                raw
            )
            pred_steer.append(
                s.cpu().numpy()
            )
            pred_speed.append(
                v.cpu().numpy()
            )

    pred_steer = np.concatenate(
        pred_steer,
        axis=0,
    )
    pred_speed = np.concatenate(
        pred_speed,
        axis=0,
    )

    steer_metrics = regression_metrics(
        pred_steer,
        y_steer,
    )

    speed_mae = float(
        np.mean(
            np.abs(
                pred_speed
                - y_speed
            )
        )
    )

    return (
        steer_metrics,
        speed_mae,
        pred_steer,
        pred_speed,
    )


def behavior_clone_actor(
    agent,
    X,
    y_steer,
    spawns,
    steps,
    device,
    epochs,
    batch_size,
    lr,
    speed_distill_weight,
    strong_weight,
    seed,
):
    actor = get_actor(agent)
    old_actor = get_old_actor(agent)

    frozen_source_actor = copy.deepcopy(
        actor
    ).to(device)
    frozen_source_actor.eval()

    actor.to(device)
    actor.train()

    train_mask, val_mask = split_mask(
        spawns,
        steps,
    )

    # Distill source PPO speed output on every demonstration state so BC
    # changes steering while preserving the already learned speed behavior.
    source_speed = np.zeros(
        (X.shape[0],),
        dtype=np.float32,
    )

    with torch.no_grad():
        for start in range(
            0,
            X.shape[0],
            1024,
        ):
            xb = torch.from_numpy(
                X[start:start + 1024]
            ).to(device)

            raw = frozen_source_actor(
                xb
            )
            _, speed = transform_actor_raw(
                raw
            )
            source_speed[
                start:start + len(speed)
            ] = speed.cpu().numpy()

    X_train = X[train_mask]
    steer_train = y_steer[train_mask]
    speed_train = source_speed[train_mask]

    train_ds = TensorDataset(
        torch.from_numpy(
            X_train
        ),
        torch.from_numpy(
            steer_train
        ),
        torch.from_numpy(
            speed_train
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        int(seed)
    )

    loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    opt = torch.optim.Adam(
        actor.parameters(),
        lr=float(lr),
    )

    best_state = None
    best_score = float("inf")
    best_epoch = 0

    X_val = X[val_mask]
    steer_val = y_steer[val_mask]
    speed_val = source_speed[val_mask]

    for epoch in range(
        1,
        int(epochs) + 1,
    ):
        actor.train()

        weighted_loss_sum = 0.0
        count = 0

        for (
            xb,
            steer_y,
            speed_y,
        ) in loader:
            xb = xb.to(device)
            steer_y = steer_y.to(device)
            speed_y = speed_y.to(device)

            raw = actor(xb)
            steer_pred, speed_pred = (
                transform_actor_raw(raw)
            )

            per_sample_steer = (
                steer_pred
                - steer_y
            ) ** 2

            weights = (
                1.0
                + float(strong_weight)
                * (
                    torch.abs(
                        steer_y
                    )
                    >= 0.15
                ).float()
            )

            steer_loss = torch.mean(
                weights
                * per_sample_steer
            )

            speed_loss = torch.mean(
                (
                    speed_pred
                    - speed_y
                ) ** 2
            )

            loss = (
                steer_loss
                + float(
                    speed_distill_weight
                )
                * speed_loss
            )

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=5.0,
            )

            opt.step()

            weighted_loss_sum += (
                float(loss.item())
                * int(xb.shape[0])
            )
            count += int(
                xb.shape[0]
            )

        (
            val_metrics,
            val_speed_mae,
            _,
            _,
        ) = evaluate_actor_arrays(
            actor=actor,
            X=X_val,
            y_steer=steer_val,
            y_speed=speed_val,
            device=device,
        )

        score = (
            float(
                val_metrics[
                    "strong_mae"
                ]
            )
            + 0.25
            * float(
                val_metrics[
                    "mae"
                ]
            )
            + 0.50
            * float(
                val_speed_mae
            )
        )

        if score < best_score:
            best_score = score
            best_epoch = int(
                epoch
            )
            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v
                in actor.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == int(epochs)
        ):
            print(
                "BC EPOCH {:03d} | loss={:.6f} | "
                "val steer mae={:.4f} strong={:.4f} sign={:.1f}% corr={:.4f} | "
                "speed_distill_mae={:.4f}".format(
                    epoch,
                    weighted_loss_sum
                    / max(count, 1),
                    val_metrics["mae"],
                    val_metrics["strong_mae"],
                    100.0
                    * val_metrics[
                        "strong_sign_acc"
                    ],
                    val_metrics["corr"],
                    val_speed_mae,
                )
            )

    if best_state is None:
        raise RuntimeError(
            "BC produced no best state."
        )

    actor.load_state_dict(
        best_state,
        strict=True,
    )
    old_actor.load_state_dict(
        best_state,
        strict=True,
    )

    actor.eval()
    old_actor.eval()

    (
        train_metrics,
        train_speed_mae,
        _,
        _,
    ) = evaluate_actor_arrays(
        actor=actor,
        X=X[train_mask],
        y_steer=y_steer[train_mask],
        y_speed=source_speed[train_mask],
        device=device,
    )

    (
        val_metrics,
        val_speed_mae,
        _,
        _,
    ) = evaluate_actor_arrays(
        actor=actor,
        X=X[val_mask],
        y_steer=y_steer[val_mask],
        y_speed=source_speed[val_mask],
        device=device,
    )

    print(
        "BC BEST | epoch={} | train steer mae={:.4f} strong={:.4f} | "
        "val steer mae={:.4f} strong={:.4f} sign={:.1f}% corr={:.4f} | "
        "val speed distill mae={:.4f}".format(
            best_epoch,
            train_metrics["mae"],
            train_metrics["strong_mae"],
            val_metrics["mae"],
            val_metrics["strong_mae"],
            100.0
            * val_metrics[
                "strong_sign_acc"
            ],
            val_metrics["corr"],
            val_speed_mae,
        )
    )

    return {
        "best_epoch": int(
            best_epoch
        ),
        "train_steer": train_metrics,
        "val_steer": val_metrics,
        "train_speed_distill_mae": float(
            train_speed_mae
        ),
        "val_speed_distill_mae": float(
            val_speed_mae
        ),
    }


def connect_and_load(
    host,
    port,
    map_path,
    timeout_s,
):
    deadline = (
        time.time()
        + float(timeout_s)
    )
    last_error = None

    while time.time() < deadline:
        try:
            client = carla.Client(
                str(host),
                int(port),
            )
            client.set_timeout(
                120.0
            )
            world = client.get_world()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    else:
        raise RuntimeError(
            "CARLA RPC not ready: {}".format(
                last_error
            )
        )

    expected_tail = (
        str(map_path)
        .split("/")[-1]
    )

    if expected_tail not in str(
        world.get_map().name
    ):
        print(
            "Loading map:",
            map_path,
        )
        world = client.load_world(
            str(map_path)
        )
        time.sleep(2.0)

    if expected_tail not in str(
        world.get_map().name
    ):
        raise RuntimeError(
            "Wrong map after load: {}".format(
                world.get_map().name
            )
        )

    return client, world


def build_env(
    client,
    world,
    episode_seconds,
):
    return CarlaEnvironmentRGBV5SingleMapStable(
        client=client,
        world=world,
        town="bc_actor_eval",
        safe_spawn_numbers=list(
            SAFE_SPAWNS
        ),
        desired_speed_mps=0.35,
        policy_speed_max_mps=0.35,
        max_episode_seconds=float(
            episode_seconds
        ),
        encoder_device="cpu",
        steer_rate_limit_per_s=3.0,
        speed_rate_limit_mps2=0.50,
        steer_deadband=0.015,
        dr_family_budget=1,
        dynamics_dr_enabled=False,
        vision_dr_enabled=False,
        sensor_dr_enabled=False,
        domain_randomization_seed=881188,
        dr_episode_probability=0.0,
        recovery_scenarios_enabled=False,
    )


def deterministic_action(
    agent,
    obs,
    device,
):
    actor = get_actor(
        agent
    )
    actor.eval()

    obs_arr = np.asarray(
        obs,
        dtype=np.float32,
    ).reshape(1, -1)

    if obs_arr.shape[1] != OBS_DIM:
        raise RuntimeError(
            "Expected obs100, got {}".format(
                obs_arr.shape
            )
        )

    with torch.no_grad():
        raw = actor(
            torch.from_numpy(
                obs_arr
            ).to(device)
        )
        steer, speed = (
            transform_actor_raw(
                raw
            )
        )

    return np.asarray(
        [
            float(
                steer.item()
            ),
            float(
                speed.item()
            ),
        ],
        dtype=np.float32,
    )


def evaluate_closed_loop(
    agent,
    env,
    device,
    episode_seconds,
):
    results = []

    for spawn in SAFE_SPAWNS:
        env.force_next_spawn_number(
            int(spawn)
        )
        obs = env.reset()

        max_steps = int(
            math.ceil(
                float(
                    episode_seconds
                )
                * float(
                    env.control_hz
                )
            )
        ) + 5

        reason = "max_steps"
        max_abs_lat = 0.0
        max_abs_steer = 0.0
        steer_abs_sum = 0.0
        speed_measured_sum = 0.0
        final_info = {}

        for step_idx in range(
            max_steps
        ):
            action = deterministic_action(
                agent,
                obs,
                device,
            )

            (
                obs,
                _reward,
                done,
                info,
            ) = env.step(
                action
            )

            steer_abs_sum += abs(
                float(action[0])
            )
            max_abs_steer = max(
                max_abs_steer,
                abs(
                    float(
                        action[0]
                    )
                ),
            )

            speed_measured_sum += float(
                info.get(
                    "imu_clean",
                    {},
                ).get(
                    "speed_mps",
                    0.0,
                )
            )

            max_abs_lat = max(
                max_abs_lat,
                abs(
                    float(
                        info.get(
                            "lateral_error_m",
                            0.0,
                        )
                    )
                ),
            )

            final_info = info

            if done:
                reason = str(
                    info.get(
                        "termination_reason",
                        "done",
                    )
                )
                break

        n = max(
            int(step_idx) + 1,
            1,
        )
        passed = bool(
            reason == "time_limit"
        )

        result = {
            "spawn": int(
                spawn
            ),
            "passed": bool(
                passed
            ),
            "reason": str(
                reason
            ),
            "steps": int(
                n
            ),
            "duration_s": (
                float(n)
                / float(
                    env.control_hz
                )
            ),
            "mean_abs_steer": float(
                steer_abs_sum / n
            ),
            "max_abs_steer": float(
                max_abs_steer
            ),
            "mean_speed_mps": float(
                speed_measured_sum / n
            ),
            "max_abs_lat_m": float(
                max_abs_lat
            ),
            "end_x": float(
                final_info.get(
                    "world_x",
                    0.0,
                )
            ),
            "end_y": float(
                final_info.get(
                    "world_y",
                    0.0,
                )
            ),
        }
        results.append(
            result
        )

        print(
            "BC PPO SPAWN {} | {} | reason={} | {:.1f}s | "
            "|steer| mean/max={:.3f}/{:.3f} | speed={:.3f} | lat max={:.3f}m".format(
                result["spawn"],
                (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                result["reason"],
                result["duration_s"],
                result["mean_abs_steer"],
                result["max_abs_steer"],
                result["mean_speed_mps"],
                result["max_abs_lat_m"],
            )
        )

    passed_count = sum(
        int(r["passed"])
        for r in results
    )

    return (
        results,
        passed_count,
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--project-root",
        default=".",
    )
    p.add_argument(
        "--dataset",
        default="",
    )
    p.add_argument(
        "--source-checkpoint",
        required=True,
    )
    p.add_argument(
        "--dest-model-name",
        default=(
            "automav5_rgb_mapvuong_"
            "stable035_bc_oracle_v1"
        ),
    )

    p.add_argument(
        "--host",
        default="localhost",
    )
    p.add_argument(
        "--port",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--map",
        default=MAP_PATH,
    )
    p.add_argument(
        "--rpc-wait-s",
        type=float,
        default=90.0,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )
    p.add_argument(
        "--strong-weight",
        type=float,
        default=4.0,
    )
    p.add_argument(
        "--speed-distill-weight",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--eval-seconds-per-spawn",
        type=float,
        default=30.0,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=881188,
    )
    p.add_argument(
        "--device",
        default="cpu",
    )
    p.add_argument(
        "--output-dir",
        default="bc_actor_results",
    )

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(
        args.seed
    )

    project_root = os.path.abspath(
        str(
            args.project_root
        )
    )

    dataset_path = str(
        args.dataset
    ).strip()

    if not dataset_path:
        dataset_path = latest_dataset(
            project_root
        )

    dataset_path = os.path.abspath(
        dataset_path
    )
    source_checkpoint = os.path.abspath(
        str(
            args.source_checkpoint
        )
    )

    if not os.path.isfile(
        dataset_path
    ):
        raise FileNotFoundError(
            dataset_path
        )
    if not os.path.isfile(
        source_checkpoint
    ):
        raise FileNotFoundError(
            source_checkpoint
        )

    data = np.load(
        dataset_path
    )
    X = np.asarray(
        data["obs"],
        dtype=np.float32,
    )
    y_steer = np.asarray(
        data["oracle_steer"],
        dtype=np.float32,
    )
    spawns = np.asarray(
        data["spawn"],
        dtype=np.int64,
    )
    steps = np.asarray(
        data["step"],
        dtype=np.int64,
    )

    if (
        X.ndim != 2
        or X.shape[1] != OBS_DIM
    ):
        raise RuntimeError(
            "Dataset obs shape must be [N,100], got {}".format(
                X.shape
            )
        )

    device = torch.device(
        str(
            args.device
        )
    )

    print("=" * 92)
    print("MAPVUONG BC -> REAL PPO ACTOR")
    print("=" * 92)
    print("Dataset          :", dataset_path)
    print("Samples          :", X.shape[0])
    print("Source checkpoint:", source_checkpoint)
    print("Destination      :", args.dest_model_name)
    print("PPO architecture : UNCHANGED")
    print("BC target steer  : oracle pure-pursuit")
    print("BC target speed  : source PPO deterministic speed (distillation)")
    print("DR / recovery    : OFF / OFF")
    print("=" * 92)

    agent = PPOAgent(
        town=str(
            args.dest_model_name
        ),
        action_std_init=0.02,
        device=str(
            args.device
        ),
    )

    loaded = agent.load(
        checkpoint_path=source_checkpoint
    )
    print(
        "LOADED SOURCE:",
        loaded,
    )

    # Runtime architecture safety check.
    actor = get_actor(
        agent
    ).to(device)

    with torch.no_grad():
        probe = actor(
            torch.from_numpy(
                X[:2]
            ).to(device)
        )

    if (
        probe.ndim != 2
        or probe.shape[1] != 2
    ):
        raise RuntimeError(
            "PPO actor does not produce raw action [N,2]: {}".format(
                tuple(
                    probe.shape
                )
            )
        )

    bc_metrics = behavior_clone_actor(
        agent=agent,
        X=X,
        y_steer=y_steer,
        spawns=spawns,
        steps=steps,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        speed_distill_weight=(
            args.speed_distill_weight
        ),
        strong_weight=(
            args.strong_weight
        ),
        seed=args.seed,
    )

    agent.set_action_std(
        0.02
    )

    # Save before CARLA eval. Source namespace is untouched because agent.town
    # was created with a new destination namespace.
    saved_checkpoint = agent.save()

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = os.path.join(
        str(
            args.output_dir
        ),
        "bc_ppo_actor_{}".format(
            stamp
        ),
    )
    if not os.path.isdir(
        run_dir
    ):
        os.makedirs(
            run_dir
        )

    meta_path = os.path.join(
        run_dir,
        "bc_metrics.json",
    )

    with open(
        meta_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "dataset": dataset_path,
                "source_checkpoint": source_checkpoint,
                "saved_checkpoint": str(
                    saved_checkpoint
                ),
                "dest_model_name": str(
                    args.dest_model_name
                ),
                "bc_metrics": bc_metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        "BC CHECKPOINT:",
        saved_checkpoint,
    )

    client = None
    env = None

    try:
        client, world = connect_and_load(
            host=args.host,
            port=args.port,
            map_path=args.map,
            timeout_s=args.rpc_wait_s,
        )

        env = build_env(
            client=client,
            world=world,
            episode_seconds=(
                args.eval_seconds_per_spawn
            ),
        )

        (
            results,
            passed_count,
        ) = evaluate_closed_loop(
            agent=agent,
            env=env,
            device=device,
            episode_seconds=(
                args.eval_seconds_per_spawn
            ),
        )

        summary_path = os.path.join(
            run_dir,
            "bc_closed_loop_summary.json",
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "saved_checkpoint": str(
                        saved_checkpoint
                    ),
                    "passed": int(
                        passed_count
                    ),
                    "total": len(
                        SAFE_SPAWNS
                    ),
                    "results": results,
                    "bc_metrics": bc_metrics,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        print("")
        print("=" * 92)
        print(
            "BC PPO RESULT | passed={}/{}".format(
                passed_count,
                len(
                    SAFE_SPAWNS
                ),
            )
        )

        if passed_count == len(
            SAFE_SPAWNS
        ):
            print(
                "VERDICT: PPO ACTOR ARCHITECTURE CAN EXPRESS THE ORACLE POLICY."
            )
            print(
                "NEXT: initialize a NEW PPO run from this BC checkpoint, then short low-std clean fine-tune."
            )
        else:
            print(
                "VERDICT: BC PPO actor is not yet closed-loop stable."
            )
            print(
                "Do NOT PPO-train yet; inspect BC validation and failed spawn metrics."
            )

        print(
            "Metrics :",
            meta_path,
        )
        print(
            "Summary :",
            summary_path,
        )
        print("=" * 92)

        return 0

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(
        main()
    )
