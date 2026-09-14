# -*- coding: utf-8 -*-
"""
MAPVUONG ORACLE -> STUDENT REPRESENTATION / IMITATION PROBE

This is NOT PPO training.

Goal
----
Test whether the deployed observation itself:
    obs100 = latent95 + speed + yaw + ax + prevSteer + prevSpeed
contains enough information to reproduce the steering action of the
privileged pure-pursuit oracle that already passed mapvuong 4/4.

Procedure
---------
1) Drive with the privileged oracle and collect (obs100, oracle_steer).
2) Train a small MLP student using ONLY obs100.
3) Evaluate that student closed-loop on the same 4 safe spawns.

Interpretation
--------------
- Student 4/4 PASS:
    representation is sufficient; PPO/reward optimization is the bottleneck.
    Next step = behavior-clone / initialize the PPO actor, then short PPO tune.
- Student has low supervised error but closed-loop fails:
    covariate shift; use DAgger-style teacher correction.
- Student has high strong-turn error:
    current latent/observation does not expose corner state reliably enough.
    Next step = inspect/retrain visual encoder or add temporal context.

No waypoint/oracle signal enters the student input.
"""

from __future__ import print_function

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from simulation.carla_connection_v5 import carla
from simulation.carla_environment_rgb_v5_singlemap_stable import (
    CarlaEnvironmentRGBV5SingleMapStable,
)
from vehicle_specs_v5 import REAL, CARLA_GEOMETRY_SCALE


MAP_PATH = "/Game/mapvuong/mapvuong"
SAFE_SPAWNS = [1, 2, 3, 4]
OBS_DIM = 100


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def clip(v, lo, hi):
    return max(lo, min(hi, float(v)))


def wrap_pi(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def connect_and_load(host, port, map_path, timeout_s):
    deadline = time.time() + float(timeout_s)
    last_error = None
    client = None

    while time.time() < deadline:
        try:
            client = carla.Client(str(host), int(port))
            client.set_timeout(120.0)
            world = client.get_world()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    else:
        raise RuntimeError(
            "CARLA RPC not ready: {}".format(last_error)
        )

    expected_tail = str(map_path).split("/")[-1]
    current = str(world.get_map().name)

    if expected_tail not in current:
        print("Loading map:", map_path)
        world = client.load_world(str(map_path))
        time.sleep(2.0)

    current = str(world.get_map().name)
    if expected_tail not in current:
        raise RuntimeError(
            "Wrong map after load: {}".format(current)
        )

    print("CARLA READY | map={}".format(current))
    return client, world


def build_env(
    client,
    world,
    episode_seconds,
    steer_rate,
    speed_slew,
    deadband,
    encoder_device,
):
    return CarlaEnvironmentRGBV5SingleMapStable(
        client=client,
        world=world,
        town="oracle_student_probe",
        safe_spawn_numbers=list(SAFE_SPAWNS),
        desired_speed_mps=0.35,
        policy_speed_max_mps=0.35,
        max_episode_seconds=float(episode_seconds),
        encoder_device=encoder_device,
        steer_rate_limit_per_s=float(steer_rate),
        speed_rate_limit_mps2=float(speed_slew),
        steer_deadband=float(deadband),
        dr_family_budget=1,
        dynamics_dr_enabled=False,
        vision_dr_enabled=False,
        sensor_dr_enabled=False,
        domain_randomization_seed=771177,
        dr_episode_probability=0.0,
        recovery_scenarios_enabled=False,
    )


def oracle_waypoint_target(env, lookahead_real_m):
    road_metrics = env.reward_manager.road_metrics
    projected_wp = road_metrics._driving_waypoint_projected()

    if projected_wp is None:
        raise RuntimeError("No projected Driving waypoint.")

    lookahead_carla_m = (
        float(lookahead_real_m)
        * float(CARLA_GEOMETRY_SCALE)
    )

    candidates = list(
        projected_wp.next(float(lookahead_carla_m))
    )

    if not candidates:
        raise RuntimeError(
            "Waypoint.next returned no candidates."
        )

    same_lane = [
        wp
        for wp in candidates
        if (
            getattr(wp, "road_id", None)
            == getattr(projected_wp, "road_id", None)
            and
            getattr(wp, "lane_id", None)
            == getattr(projected_wp, "lane_id", None)
        )
    ]
    pool = same_lane if same_lane else candidates

    current_road_yaw = math.radians(
        float(projected_wp.transform.rotation.yaw)
    )

    def tangent_jump(wp):
        yaw = math.radians(
            float(wp.transform.rotation.yaw)
        )
        return abs(
            wrap_pi(yaw - current_road_yaw)
        )

    target_wp = min(pool, key=tangent_jump)

    vehicle_tf = env.vehicle.get_transform()
    vehicle_loc = vehicle_tf.location
    target_loc = target_wp.transform.location

    dx = float(target_loc.x - vehicle_loc.x)
    dy = float(target_loc.y - vehicle_loc.y)

    distance_carla_m = math.sqrt(dx * dx + dy * dy)
    distance_real_m = (
        distance_carla_m
        / max(float(CARLA_GEOMETRY_SCALE), 1e-9)
    )

    target_bearing = math.atan2(dy, dx)
    vehicle_yaw = math.radians(
        float(vehicle_tf.rotation.yaw)
    )

    alpha = wrap_pi(
        target_bearing - vehicle_yaw
    )

    return float(alpha), float(distance_real_m)


def pure_pursuit_steer_unit(alpha_rad, lookahead_distance_m):
    wheelbase = float(REAL.wheelbase_m)
    max_steer_rad = math.radians(
        float(REAL.max_steer_angle_deg)
    )

    delta_rad = math.atan2(
        2.0
        * wheelbase
        * math.sin(float(alpha_rad)),
        max(float(lookahead_distance_m), 0.05),
    )

    steer_unit = (
        delta_rad
        / max(max_steer_rad, 1e-9)
    )

    return clip(steer_unit, -1.0, 1.0)


def collect_oracle_dataset(
    env,
    lookahead_m,
    speed_mps,
    seconds_per_spawn,
):
    obs_rows = []
    steer_rows = []
    spawn_rows = []
    step_rows = []
    strong_rows = []

    speed_unit = clip(
        float(speed_mps)
        / float(env.policy_speed_max_mps),
        0.0,
        1.0,
    )

    old_limit = float(env.max_episode_seconds)
    env.max_episode_seconds = float(seconds_per_spawn)

    for spawn in SAFE_SPAWNS:
        env.force_next_spawn_number(int(spawn))
        obs = env.reset()

        max_steps = int(
            math.ceil(
                float(seconds_per_spawn)
                * float(env.control_hz)
            )
        ) + 5

        collected = 0
        reason = "max_steps"

        for step_idx in range(max_steps):
            alpha, ld = oracle_waypoint_target(
                env,
                lookahead_m,
            )
            target_steer = pure_pursuit_steer_unit(
                alpha,
                ld,
            )

            obs_arr = np.asarray(
                obs,
                dtype=np.float32,
            ).reshape(-1)

            if obs_arr.shape[0] != OBS_DIM:
                raise RuntimeError(
                    "Expected obs100, got {}".format(
                        obs_arr.shape
                    )
                )

            obs_rows.append(obs_arr.copy())
            steer_rows.append(float(target_steer))
            spawn_rows.append(int(spawn))
            step_rows.append(int(step_idx))
            strong_rows.append(
                int(abs(float(target_steer)) >= 0.15)
            )

            action = np.asarray(
                [
                    float(target_steer),
                    float(speed_unit),
                ],
                dtype=np.float32,
            )

            (
                obs,
                _reward,
                done,
                info,
            ) = env.step(action)

            collected += 1

            if done:
                reason = str(
                    info.get(
                        "termination_reason",
                        "done",
                    )
                )
                break

        print(
            "COLLECT SPAWN {} | samples={} | reason={}".format(
                spawn,
                collected,
                reason,
            )
        )

    env.max_episode_seconds = old_limit

    X = np.asarray(obs_rows, dtype=np.float32)
    y = np.asarray(steer_rows, dtype=np.float32)
    spawns = np.asarray(spawn_rows, dtype=np.int64)
    steps = np.asarray(step_rows, dtype=np.int64)
    strong = np.asarray(strong_rows, dtype=np.int64)

    print(
        "DATASET | n={} | strong_turn={:.1f}% | target |steer| mean={:.3f} max={:.3f}".format(
            int(X.shape[0]),
            100.0 * float(strong.mean()) if strong.size else 0.0,
            float(np.mean(np.abs(y))) if y.size else 0.0,
            float(np.max(np.abs(y))) if y.size else 0.0,
        )
    )

    return X, y, spawns, steps, strong


class SteeringStudent(nn.Module):
    def __init__(self):
        super(SteeringStudent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def block_split(spawns, steps):
    """
    Validation uses every 5th 2-second block within each spawn.
    This avoids a fully random frame split with near-duplicate camera frames.
    """
    block_id = steps // 100
    val_mask = (block_id % 5) == 4
    train_mask = ~val_mask

    if not np.any(val_mask):
        raise RuntimeError("Validation split is empty.")
    if not np.any(train_mask):
        raise RuntimeError("Training split is empty.")

    return train_mask, val_mask


def metrics_np(pred, target):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.float32).reshape(-1)

    err = np.abs(pred - target)
    strong = np.abs(target) >= 0.15

    result = {
        "mae": float(np.mean(err)),
        "rmse": float(
            np.sqrt(
                np.mean(
                    (pred - target) ** 2
                )
            )
        ),
        "strong_count": int(np.sum(strong)),
        "strong_mae": (
            float(np.mean(err[strong]))
            if np.any(strong)
            else 0.0
        ),
        "strong_sign_acc": 0.0,
        "corr": 0.0,
    }

    if np.any(strong):
        result["strong_sign_acc"] = float(
            np.mean(
                np.sign(pred[strong])
                == np.sign(target[strong])
            )
        )

    if pred.size >= 2:
        pstd = float(np.std(pred))
        tstd = float(np.std(target))
        if pstd > 1e-8 and tstd > 1e-8:
            result["corr"] = float(
                np.corrcoef(pred, target)[0, 1]
            )

    return result


def train_student(
    X,
    y,
    spawns,
    steps,
    device,
    epochs,
    batch_size,
    lr,
    seed,
):
    train_mask, val_mask = block_split(
        spawns,
        steps,
    )

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    mean = np.mean(
        X_train,
        axis=0,
        keepdims=True,
    ).astype(np.float32)
    std = np.std(
        X_train,
        axis=0,
        keepdims=True,
    ).astype(np.float32)
    std = np.maximum(std, 1e-5)

    X_train_n = (
        (X_train - mean)
        / std
    ).astype(np.float32)
    X_val_n = (
        (X_val - mean)
        / std
    ).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n),
        torch.from_numpy(y_train),
    )

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    model = SteeringStudent().to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
    )

    best_state = None
    best_strong_mae = float("inf")
    best_epoch = 0

    Xv = torch.from_numpy(X_val_n).to(device)
    yv = torch.from_numpy(y_val).to(device)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = 0.0
        count = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            # Strong-turn samples matter more; otherwise long straights dominate.
            weight = (
                1.0
                + 4.0
                * (torch.abs(yb) >= 0.15).float()
            )

            loss = torch.mean(
                weight
                * (pred - yb) ** 2
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * int(xb.shape[0])
            count += int(xb.shape[0])

        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy()

        vm = metrics_np(
            pv,
            y_val,
        )

        criterion = float(
            vm["strong_mae"]
            if vm["strong_count"] > 0
            else vm["mae"]
        )

        if criterion < best_strong_mae:
            best_strong_mae = criterion
            best_epoch = int(epoch)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == int(epochs)
        ):
            print(
                "STUDENT EPOCH {:03d} | train_wmse={:.6f} | "
                "val_mae={:.4f} strong_mae={:.4f} sign={:.1f}% corr={:.3f}".format(
                    epoch,
                    loss_sum / max(count, 1),
                    vm["mae"],
                    vm["strong_mae"],
                    100.0 * vm["strong_sign_acc"],
                    vm["corr"],
                )
            )

    if best_state is None:
        raise RuntimeError("Student training produced no best state.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_pred = model(
            torch.from_numpy(
                (
                    (X_train - mean)
                    / std
                ).astype(np.float32)
            ).to(device)
        ).cpu().numpy()

        val_pred = model(
            torch.from_numpy(X_val_n).to(device)
        ).cpu().numpy()

    train_metrics = metrics_np(
        train_pred,
        y_train,
    )
    val_metrics = metrics_np(
        val_pred,
        y_val,
    )

    print(
        "BEST STUDENT | epoch={} | train_mae={:.4f} strong={:.4f} | "
        "val_mae={:.4f} strong={:.4f} sign={:.1f}% corr={:.3f}".format(
            best_epoch,
            train_metrics["mae"],
            train_metrics["strong_mae"],
            val_metrics["mae"],
            val_metrics["strong_mae"],
            100.0 * val_metrics["strong_sign_acc"],
            val_metrics["corr"],
        )
    )

    return (
        model,
        mean.reshape(-1).astype(np.float32),
        std.reshape(-1).astype(np.float32),
        train_metrics,
        val_metrics,
        best_epoch,
    )


def evaluate_student_closed_loop(
    env,
    model,
    obs_mean,
    obs_std,
    device,
    speed_mps,
    eval_seconds,
    lookahead_m,
):
    speed_unit = clip(
        float(speed_mps)
        / float(env.policy_speed_max_mps),
        0.0,
        1.0,
    )

    old_limit = float(env.max_episode_seconds)
    env.max_episode_seconds = float(eval_seconds)

    results = []

    for spawn in SAFE_SPAWNS:
        env.force_next_spawn_number(int(spawn))
        obs = env.reset()

        max_steps = int(
            math.ceil(
                float(eval_seconds)
                * float(env.control_hz)
            )
        ) + 5

        reason = "max_steps"
        abs_teacher_err_sum = 0.0
        strong_teacher_err_sum = 0.0
        strong_count = 0
        abs_student_sum = 0.0
        max_abs_student = 0.0
        max_abs_lat = 0.0
        final_info = {}

        for step_idx in range(max_steps):
            obs_arr = np.asarray(
                obs,
                dtype=np.float32,
            ).reshape(-1)

            norm = (
                (obs_arr - obs_mean)
                / obs_std
            ).astype(np.float32)

            with torch.no_grad():
                student_steer = float(
                    model(
                        torch.from_numpy(norm)
                        .unsqueeze(0)
                        .to(device)
                    ).item()
                )

            alpha, ld = oracle_waypoint_target(
                env,
                lookahead_m,
            )
            oracle_steer = pure_pursuit_steer_unit(
                alpha,
                ld,
            )

            err = abs(
                student_steer
                - oracle_steer
            )
            abs_teacher_err_sum += err

            if abs(oracle_steer) >= 0.15:
                strong_teacher_err_sum += err
                strong_count += 1

            abs_student_sum += abs(student_steer)
            max_abs_student = max(
                max_abs_student,
                abs(student_steer),
            )

            action = np.asarray(
                [
                    float(student_steer),
                    float(speed_unit),
                ],
                dtype=np.float32,
            )

            (
                obs,
                _reward,
                done,
                info,
            ) = env.step(action)

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

        n = max(step_idx + 1, 1)
        passed = bool(reason == "time_limit")

        result = {
            "spawn": int(spawn),
            "passed": bool(passed),
            "reason": str(reason),
            "steps": int(n),
            "duration_s": float(n) / float(env.control_hz),
            "mean_abs_student_steer": float(
                abs_student_sum / n
            ),
            "max_abs_student_steer": float(
                max_abs_student
            ),
            "mean_teacher_error": float(
                abs_teacher_err_sum / n
            ),
            "strong_teacher_error": float(
                strong_teacher_err_sum
                / max(strong_count, 1)
            ),
            "strong_count": int(strong_count),
            "max_abs_lat_m": float(max_abs_lat),
            "end_x": float(
                final_info.get("world_x", 0.0)
            ),
            "end_y": float(
                final_info.get("world_y", 0.0)
            ),
        }
        results.append(result)

        print(
            "STUDENT SPAWN {} | {} | reason={} | {:.1f}s | "
            "|steer| mean/max={:.3f}/{:.3f} | teacher err={:.3f} "
            "strong={:.3f} | lat max={:.3f}m".format(
                result["spawn"],
                "PASS" if passed else "FAIL",
                result["reason"],
                result["duration_s"],
                result["mean_abs_student_steer"],
                result["max_abs_student_steer"],
                result["mean_teacher_error"],
                result["strong_teacher_error"],
                result["max_abs_lat_m"],
            )
        )

    env.max_episode_seconds = old_limit

    passed_count = sum(
        int(r["passed"])
        for r in results
    )
    return results, passed_count


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--map", default=MAP_PATH)
    p.add_argument("--rpc-wait-s", type=float, default=90.0)

    p.add_argument("--lookahead-m", type=float, default=0.50)
    p.add_argument("--speed-mps", type=float, default=0.20)

    p.add_argument(
        "--collect-seconds-per-spawn",
        type=float,
        default=45.0,
    )
    p.add_argument(
        "--eval-seconds-per-spawn",
        type=float,
        default=30.0,
    )

    p.add_argument("--steer-rate", type=float, default=3.0)
    p.add_argument("--speed-slew", type=float, default=0.50)
    p.add_argument("--deadband", type=float, default=0.015)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--device", default="cpu")
    p.add_argument("--encoder-device", default="cpu")
    p.add_argument("--seed", type=int, default=771177)
    p.add_argument(
        "--output-dir",
        default="oracle_student_results",
    )

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if not (
        0.0 < float(args.speed_mps) <= 0.35
    ):
        raise ValueError(
            "--speed-mps must be in (0, 0.35]."
        )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = os.path.join(
        str(args.output_dir),
        "student_probe_{}".format(stamp),
    )
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    device = torch.device(str(args.device))

    print("=" * 88)
    print("MAPVUONG ORACLE -> STUDENT PROBE")
    print("=" * 88)
    print("Student input     : obs100 ONLY")
    print("Privileged input  : NONE")
    print("Oracle            : pure-pursuit waypoint teacher")
    print("Collect           : {:.1f}s x 4 spawns".format(args.collect_seconds_per_spawn))
    print("Closed-loop eval  : {:.1f}s x 4 spawns".format(args.eval_seconds_per_spawn))
    print("Speed             : {:.3f} m/s".format(args.speed_mps))
    print("Lookahead         : {:.3f} m".format(args.lookahead_m))
    print("Steer rate        : {:.3f} /s".format(args.steer_rate))
    print("DR / recovery     : OFF / OFF")
    print("=" * 88)

    client = None
    env = None

    try:
        client, world = connect_and_load(
            args.host,
            args.port,
            args.map,
            args.rpc_wait_s,
        )

        env = build_env(
            client=client,
            world=world,
            episode_seconds=max(
                float(args.collect_seconds_per_spawn),
                float(args.eval_seconds_per_spawn),
            ),
            steer_rate=args.steer_rate,
            speed_slew=args.speed_slew,
            deadband=args.deadband,
            encoder_device=args.encoder_device,
        )

        (
            X,
            y,
            spawns,
            steps,
            strong,
        ) = collect_oracle_dataset(
            env=env,
            lookahead_m=args.lookahead_m,
            speed_mps=args.speed_mps,
            seconds_per_spawn=args.collect_seconds_per_spawn,
        )

        dataset_path = os.path.join(
            run_dir,
            "oracle_obs100_dataset.npz",
        )

        np.savez_compressed(
            dataset_path,
            obs=X,
            oracle_steer=y,
            spawn=spawns,
            step=steps,
            strong=strong,
        )

        (
            model,
            obs_mean,
            obs_std,
            train_metrics,
            val_metrics,
            best_epoch,
        ) = train_student(
            X=X,
            y=y,
            spawns=spawns,
            steps=steps,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
        )

        model_path = os.path.join(
            run_dir,
            "student_steer_probe.pt",
        )

        torch.save(
            {
                "model_state": model.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
                "best_epoch": int(best_epoch),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "config": vars(args),
            },
            model_path,
        )

        print("")
        print("=" * 88)
        print("CLOSED-LOOP STUDENT EVALUATION")
        print("=" * 88)

        (
            eval_results,
            passed_count,
        ) = evaluate_student_closed_loop(
            env=env,
            model=model,
            obs_mean=obs_mean,
            obs_std=obs_std,
            device=device,
            speed_mps=args.speed_mps,
            eval_seconds=args.eval_seconds_per_spawn,
            lookahead_m=args.lookahead_m,
        )

        summary = {
            "dataset_samples": int(X.shape[0]),
            "strong_turn_fraction": float(
                np.mean(strong)
            ),
            "best_epoch": int(best_epoch),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "closed_loop_passed": int(
                passed_count
            ),
            "closed_loop_total": int(
                len(SAFE_SPAWNS)
            ),
            "closed_loop": eval_results,
        }

        summary_path = os.path.join(
            run_dir,
            "student_probe_summary.json",
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                sort_keys=True,
            )

        print("")
        print("=" * 88)
        print(
            "STUDENT RESULT | passed={}/{}".format(
                passed_count,
                len(SAFE_SPAWNS),
            )
        )
        print(
            "VAL | mae={:.4f} | strong_mae={:.4f} | strong_sign={:.1f}% | corr={:.3f}".format(
                val_metrics["mae"],
                val_metrics["strong_mae"],
                100.0 * val_metrics["strong_sign_acc"],
                val_metrics["corr"],
            )
        )

        if passed_count == len(SAFE_SPAWNS):
            print(
                "VERDICT: obs100/latent95 IS SUFFICIENT for the mapvuong steering task."
            )
            print(
                "NEXT: behavior-clone/initialize the PPO actor from oracle demonstrations, then short PPO fine-tune."
            )
        elif (
            val_metrics["strong_mae"] <= 0.08
            and val_metrics["strong_sign_acc"] >= 0.95
        ):
            print(
                "VERDICT: supervised mapping is good but closed-loop fails -> covariate shift."
            )
            print(
                "NEXT: DAgger-style teacher correction, not more reward-only PPO."
            )
        else:
            print(
                "VERDICT: obs100 cannot reliably reproduce strong oracle steering yet."
            )
            print(
                "NEXT: inspect latent/camera corner separability and temporal context."
            )

        print("Dataset :", dataset_path)
        print("Model   :", model_path)
        print("Summary :", summary_path)
        print("=" * 88)

        return 0

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
