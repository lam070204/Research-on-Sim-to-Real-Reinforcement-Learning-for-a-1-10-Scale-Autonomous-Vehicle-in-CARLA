# -*- coding: utf-8 -*-
"""
MAPVUONG ORACLE CORNER TEST

Purpose
-------
Bypass PPO entirely and drive the exact STABLE035 environment with a
privileged pure-pursuit controller using CARLA waypoints.

This isolates:
- vehicle physics / steering authority
- 3.0/s stable steering rate limiter
- speed controller
- map / waypoint geometry
- wheel-based offroad termination

It keeps:
- same CARLA vehicle + wheel geometry
- same StableActionAdapterV5
- same ActionControllerV5
- same offroad detector
- same 4 safe spawns
- DR OFF, recovery OFF

PPO/VAE observation is still built by the environment, but PPO does not
choose actions in this test.
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

from simulation.carla_connection_v5 import carla
from simulation.carla_environment_rgb_v5_singlemap_stable import (
    CarlaEnvironmentRGBV5SingleMapStable,
)
from vehicle_specs_v5 import REAL, CARLA_GEOMETRY_SCALE


MAP_PATH = "/Game/mapvuong/mapvuong"
SAFE_SPAWNS = [1, 2, 3, 4]


def clip(v, lo, hi):
    return max(lo, min(hi, float(v)))


def wrap_pi(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def waypoint_target(env, lookahead_real_m):
    """
    Find a forward Driving waypoint and return a pure-pursuit target.

    Selection intentionally follows the same branch preference used by the
    v1.4 reward lookahead:
      1) same road/lane when available
      2) otherwise smallest tangent discontinuity
    """
    road_metrics = env.reward_manager.road_metrics
    projected_wp = road_metrics._driving_waypoint_projected()

    if projected_wp is None:
        raise RuntimeError("No projected Driving waypoint.")

    lookahead_carla_m = (
        float(lookahead_real_m)
        * float(CARLA_GEOMETRY_SCALE)
    )

    candidates = list(
        projected_wp.next(
            float(lookahead_carla_m)
        )
    )

    if not candidates:
        raise RuntimeError(
            "Waypoint.next({:.3f} CARLA m) returned no candidates.".format(
                lookahead_carla_m
            )
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
        float(
            projected_wp.transform.rotation.yaw
        )
    )

    def tangent_jump(wp):
        yaw = math.radians(
            float(
                wp.transform.rotation.yaw
            )
        )
        return abs(
            wrap_pi(
                yaw - current_road_yaw
            )
        )

    target_wp = min(
        pool,
        key=tangent_jump,
    )

    vehicle_tf = env.vehicle.get_transform()
    vehicle_loc = vehicle_tf.location
    target_loc = target_wp.transform.location

    dx = float(target_loc.x - vehicle_loc.x)
    dy = float(target_loc.y - vehicle_loc.y)

    distance_carla_m = math.sqrt(
        dx * dx + dy * dy
    )
    distance_real_m = (
        distance_carla_m
        / max(float(CARLA_GEOMETRY_SCALE), 1e-9)
    )

    target_bearing = math.atan2(
        dy,
        dx,
    )
    vehicle_yaw = math.radians(
        float(vehicle_tf.rotation.yaw)
    )

    # CARLA/UE: +Y and +yaw are to the right, and positive steer turns right.
    alpha = wrap_pi(
        target_bearing
        - vehicle_yaw
    )

    return {
        "projected_wp": projected_wp,
        "target_wp": target_wp,
        "distance_real_m": float(distance_real_m),
        "alpha_rad": float(alpha),
        "target_bearing_rad": float(target_bearing),
        "vehicle_yaw_rad": float(vehicle_yaw),
    }


def pure_pursuit_steer_unit(
    alpha_rad,
    lookahead_distance_m,
):
    """
    Bicycle-model pure pursuit:
        delta = atan2(2 L sin(alpha), Ld)

    Normalize physical front-wheel steering angle to PPO/CARLA steer [-1,1].
    """
    wheelbase = float(
        REAL.wheelbase_m
    )
    max_steer_rad = math.radians(
        float(
            REAL.max_steer_angle_deg
        )
    )

    ld = max(
        float(lookahead_distance_m),
        0.05,
    )

    delta_rad = math.atan2(
        2.0
        * wheelbase
        * math.sin(float(alpha_rad)),
        ld,
    )

    steer_unit = (
        delta_rad
        / max(max_steer_rad, 1e-9)
    )

    return (
        clip(
            steer_unit,
            -1.0,
            1.0,
        ),
        float(delta_rad),
    )


def connect_and_load(
    host,
    port,
    map_path,
    timeout_s,
):
    deadline = time.time() + float(timeout_s)
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

    current = str(
        world.get_map().name
    )

    expected_tail = str(
        map_path
    ).split("/")[-1]

    if expected_tail not in current:
        print(
            "Loading map:",
            map_path,
        )
        world = client.load_world(
            str(map_path)
        )
        time.sleep(2.0)

    current = str(
        world.get_map().name
    )
    if expected_tail not in current:
        raise RuntimeError(
            "Wrong map after load: {}".format(
                current
            )
        )

    print(
        "CARLA READY | map={}".format(
            current
        )
    )
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
        town="oracle_mapvuong",
        safe_spawn_numbers=list(
            SAFE_SPAWNS
        ),
        desired_speed_mps=0.35,
        policy_speed_max_mps=0.35,
        max_episode_seconds=float(
            episode_seconds
        ),
        encoder_device=encoder_device,
        steer_rate_limit_per_s=float(
            steer_rate
        ),
        speed_rate_limit_mps2=float(
            speed_slew
        ),
        steer_deadband=float(
            deadband
        ),
        dr_family_budget=1,
        dynamics_dr_enabled=False,
        vision_dr_enabled=False,
        sensor_dr_enabled=False,
        domain_randomization_seed=660066,
        dr_episode_probability=0.0,
        recovery_scenarios_enabled=False,
    )


def run_episode(
    env,
    spawn_number,
    lookahead_m,
    speed_cmd_mps,
    step_writer,
):
    env.force_next_spawn_number(
        int(spawn_number)
    )
    env.reset()

    speed_unit = clip(
        float(speed_cmd_mps)
        / float(env.policy_speed_max_mps),
        0.0,
        1.0,
    )

    max_steps = int(
        math.ceil(
            float(env.max_episode_seconds)
            * float(env.control_hz)
        )
    ) + 10

    reward_sum = 0.0
    progress_sum = 0.0
    speed_sum = 0.0
    lat_sum = 0.0
    heading_now_sum = 0.0
    heading_preview_sum = 0.0
    requested_abs_sum = 0.0
    applied_abs_sum = 0.0
    limiter_hits = 0

    max_abs_lat = 0.0
    max_abs_heading_now_deg = 0.0
    max_abs_heading_preview_deg = 0.0
    max_abs_required_steer = 0.0
    max_abs_applied_steer = 0.0

    reason = "max_steps"
    done = False
    last_info = {}

    for step_index in range(1, max_steps + 1):
        target = waypoint_target(
            env,
            lookahead_m,
        )

        required_steer, delta_rad = (
            pure_pursuit_steer_unit(
                alpha_rad=target[
                    "alpha_rad"
                ],
                lookahead_distance_m=target[
                    "distance_real_m"
                ],
            )
        )

        action = np.asarray(
            [
                float(required_steer),
                float(speed_unit),
            ],
            dtype=np.float32,
        )

        (
            _obs,
            reward,
            done,
            info,
        ) = env.step(
            action
        )

        control = dict(
            info.get(
                "control",
                {},
            )
        )

        requested_steer = float(
            info.get(
                "stable_filtered_steer_cmd",
                required_steer,
            )
        )
        applied_steer = float(
            control.get(
                "actual_steer_cmd",
                requested_steer,
            )
        )

        limiter_hit = bool(
            control.get(
                "steer_rate_limiter_hit",
                False,
            )
        )

        speed_mps = float(
            info.get(
                "imu_clean",
                {},
            ).get(
                "speed_mps",
                0.0,
            )
        )
        lateral_error_m = float(
            info.get(
                "lateral_error_m",
                0.0,
            )
        )
        heading_now_rad = float(
            info.get(
                "heading_error_rad",
                0.0,
            )
        )
        heading_preview_rad = float(
            info.get(
                "preview_heading_error_rad",
                -target["alpha_rad"],
            )
        )
        yaw_rate = float(
            info.get(
                "imu_clean",
                {},
            ).get(
                "yaw_rate_rps",
                info.get(
                    "imu_clean",
                    {},
                ).get(
                    "yaw_rate",
                    0.0,
                ),
            )
        )

        reward_sum += float(reward)
        progress_sum += float(
            info.get(
                "forward_progress_m",
                0.0,
            )
        )
        speed_sum += speed_mps
        lat_sum += abs(
            lateral_error_m
        )
        heading_now_sum += abs(
            heading_now_rad
        )
        heading_preview_sum += abs(
            heading_preview_rad
        )
        requested_abs_sum += abs(
            required_steer
        )
        applied_abs_sum += abs(
            applied_steer
        )
        limiter_hits += int(
            limiter_hit
        )

        max_abs_lat = max(
            max_abs_lat,
            abs(lateral_error_m),
        )
        max_abs_heading_now_deg = max(
            max_abs_heading_now_deg,
            abs(math.degrees(heading_now_rad)),
        )
        max_abs_heading_preview_deg = max(
            max_abs_heading_preview_deg,
            abs(math.degrees(heading_preview_rad)),
        )
        max_abs_required_steer = max(
            max_abs_required_steer,
            abs(required_steer),
        )
        max_abs_applied_steer = max(
            max_abs_applied_steer,
            abs(applied_steer),
        )

        step_writer.writerow(
            {
                "spawn": int(
                    spawn_number
                ),
                "step": int(
                    step_index
                ),
                "time_s": (
                    float(step_index)
                    / float(env.control_hz)
                ),
                "world_x": float(
                    info.get(
                        "world_x",
                        0.0,
                    )
                ),
                "world_y": float(
                    info.get(
                        "world_y",
                        0.0,
                    )
                ),
                "speed_mps": speed_mps,
                "speed_cmd_mps": float(
                    speed_cmd_mps
                ),
                "lookahead_m": float(
                    target[
                        "distance_real_m"
                    ]
                ),
                "alpha_deg": math.degrees(
                    float(
                        target[
                            "alpha_rad"
                        ]
                    )
                ),
                "required_delta_deg": math.degrees(
                    float(
                        delta_rad
                    )
                ),
                "required_steer_unit": float(
                    required_steer
                ),
                "requested_steer_unit": float(
                    requested_steer
                ),
                "applied_steer_unit": float(
                    applied_steer
                ),
                "steer_rate_limiter_hit": int(
                    limiter_hit
                ),
                "lateral_error_m": lateral_error_m,
                "heading_now_deg": math.degrees(
                    heading_now_rad
                ),
                "heading_preview_deg": math.degrees(
                    heading_preview_rad
                ),
                "yaw_rate_rps": yaw_rate,
                "reward": float(
                    reward
                ),
                "offroad": int(
                    bool(
                        info.get(
                            "offroad",
                            False,
                        )
                    )
                ),
                "done": int(
                    bool(done)
                ),
                "termination_reason": str(
                    info.get(
                        "termination_reason",
                        "",
                    )
                ),
            }
        )

        last_info = info

        if done:
            reason = str(
                info.get(
                    "termination_reason",
                    "done",
                )
            )
            break

    n = max(
        int(step_index),
        1,
    )

    passed = bool(
        reason == "time_limit"
    )

    return {
        "spawn": int(
            spawn_number
        ),
        "passed": int(
            passed
        ),
        "reason": str(
            reason
        ),
        "steps": int(
            step_index
        ),
        "duration_s": (
            float(step_index)
            / float(env.control_hz)
        ),
        "reward_sum": float(
            reward_sum
        ),
        "progress_sum_m": float(
            progress_sum
        ),
        "mean_speed_mps": float(
            speed_sum / n
        ),
        "mean_abs_lat_m": float(
            lat_sum / n
        ),
        "max_abs_lat_m": float(
            max_abs_lat
        ),
        "mean_abs_heading_now_deg": math.degrees(
            heading_now_sum / n
        ),
        "max_abs_heading_now_deg": float(
            max_abs_heading_now_deg
        ),
        "mean_abs_heading_preview_deg": math.degrees(
            heading_preview_sum / n
        ),
        "max_abs_heading_preview_deg": float(
            max_abs_heading_preview_deg
        ),
        "mean_abs_required_steer": float(
            requested_abs_sum / n
        ),
        "max_abs_required_steer": float(
            max_abs_required_steer
        ),
        "mean_abs_applied_steer": float(
            applied_abs_sum / n
        ),
        "max_abs_applied_steer": float(
            max_abs_applied_steer
        ),
        "limiter_hit_pct": float(
            100.0
            * float(limiter_hits)
            / float(n)
        ),
        "end_x": float(
            last_info.get(
                "world_x",
                0.0,
            )
        ),
        "end_y": float(
            last_info.get(
                "world_y",
                0.0,
            )
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        default="localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--map",
        default=MAP_PATH,
    )
    parser.add_argument(
        "--rpc-wait-s",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--lookahead-m",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--steer-rate",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--speed-slew",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=0.015,
    )
    parser.add_argument(
        "--encoder-device",
        default="cpu",
    )
    parser.add_argument(
        "--output-dir",
        default="oracle_results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (
        0.0 < float(args.speed_mps) <= 0.35
    ):
        raise ValueError(
            "--speed-mps must be in (0, 0.35]."
        )
    if float(args.lookahead_m) <= 0.0:
        raise ValueError(
            "--lookahead-m must be > 0."
        )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = os.path.join(
        str(args.output_dir),
        "mapvuong_oracle_{}".format(
            stamp
        ),
    )
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    step_csv = os.path.join(
        run_dir,
        "oracle_steps.csv",
    )
    summary_csv = os.path.join(
        run_dir,
        "oracle_summary.csv",
    )

    client = None
    env = None
    summaries = []

    print("=" * 78)
    print("MAPVUONG ORACLE CORNER TEST")
    print("=" * 78)
    print("PPO             : BYPASSED")
    print("Map             :", args.map)
    print("Spawns          :", SAFE_SPAWNS)
    print("Controller      : pure pursuit (privileged CARLA waypoint)")
    print("Lookahead       : {:.3f} m".format(args.lookahead_m))
    print("Speed command   : {:.3f} m/s".format(args.speed_mps))
    print("Steer rate      : {:.3f} /s".format(args.steer_rate))
    print("Episode limit   : {:.1f} s/spawn".format(args.episode_seconds))
    print("Pass criterion  : survives to time_limit")
    print("DR / recovery   : OFF / OFF")
    print("=" * 78)

    step_fields = [
        "spawn",
        "step",
        "time_s",
        "world_x",
        "world_y",
        "speed_mps",
        "speed_cmd_mps",
        "lookahead_m",
        "alpha_deg",
        "required_delta_deg",
        "required_steer_unit",
        "requested_steer_unit",
        "applied_steer_unit",
        "steer_rate_limiter_hit",
        "lateral_error_m",
        "heading_now_deg",
        "heading_preview_deg",
        "yaw_rate_rps",
        "reward",
        "offroad",
        "done",
        "termination_reason",
    ]

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
            episode_seconds=args.episode_seconds,
            steer_rate=args.steer_rate,
            speed_slew=args.speed_slew,
            deadband=args.deadband,
            encoder_device=args.encoder_device,
        )

        with open(
            step_csv,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=step_fields,
            )
            writer.writeheader()

            for spawn_number in SAFE_SPAWNS:
                result = run_episode(
                    env=env,
                    spawn_number=spawn_number,
                    lookahead_m=args.lookahead_m,
                    speed_cmd_mps=args.speed_mps,
                    step_writer=writer,
                )
                summaries.append(
                    result
                )

                print(
                    "SPAWN {} | {} | reason={} | steps={} ({:.1f}s) | "
                    "speed={:.3f} | progress={:+.2f}m | lat max={:.3f}m | "
                    "headingPreview max={:.1f}deg | steer req/applied max={:.3f}/{:.3f} | "
                    "limiterHit={:.1f}%".format(
                        result["spawn"],
                        (
                            "PASS"
                            if result["passed"]
                            else "FAIL"
                        ),
                        result["reason"],
                        result["steps"],
                        result["duration_s"],
                        result["mean_speed_mps"],
                        result["progress_sum_m"],
                        result["max_abs_lat_m"],
                        result["max_abs_heading_preview_deg"],
                        result["max_abs_required_steer"],
                        result["max_abs_applied_steer"],
                        result["limiter_hit_pct"],
                    )
                )

        if summaries:
            fields = list(
                summaries[0].keys()
            )
            with open(
                summary_csv,
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fields,
                )
                writer.writeheader()
                for row in summaries:
                    writer.writerow(
                        row
                    )

        passed_count = sum(
            int(row["passed"])
            for row in summaries
        )

        print("")
        print("=" * 78)
        print(
            "ORACLE RESULT | passed={}/{}".format(
                passed_count,
                len(SAFE_SPAWNS),
            )
        )

        if passed_count == len(
            SAFE_SPAWNS
        ):
            print(
                "VERDICT: PHYSICS + MAP + STEER RATE ARE CAPABLE."
            )
            print(
                "Next target: RL/reward/observation learning, not CARLA dynamics."
            )
        else:
            print(
                "VERDICT: ORACLE ALSO FAILS ONE OR MORE SPAWNS."
            )
            print(
                "Next target: steering authority / controller geometry / map topology before more PPO training."
            )

        print(
            "Summary CSV:",
            summary_csv,
        )
        print(
            "Step CSV   :",
            step_csv,
        )
        print("=" * 78)

        # Return nonzero only for script/runtime failure; oracle FAIL is a valid diagnostic.
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
