# -*- coding: utf-8 -*-
"""
GĐ7 - Freeze Guard V3
=====================

Mục tiêu:
- Khóa các phần đã hoàn tất của GĐ4-GĐ6 để project không bị "trôi config".
- KHÔNG giả vờ rằng longitudinal đã hoàn tất:
    REAL ax                  = PENDING
    longitudinal nominal     = PROVISIONAL
    longitudinal DR ranges   = PROVISIONAL
- Cho phép sau này mở đúng phần longitudinal để tune lại, rồi freeze manifest mới.

Dùng:
    python .\gd7_freeze_guard.py --freeze
    python .\gd7_freeze_guard.py --verify

--freeze:
    Tạo gd7_freeze_manifest.json từ trạng thái project hiện tại.

--verify:
    So file SHA256 + semantic constants với manifest.
    Nếu phần đã FROZEN bị đổi -> FAIL.
    Nếu longitudinal provisional thay đổi -> chỉ WARNING, không FAIL.

Python 3.7 compatible.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "gd7_freeze_manifest.json"


# Whole-file freeze: these should not need to change when REAL ax is fixed.
FROZEN_FILES = [
    "vehicle_specs_v3.py",
    "simulation/simulation_settings_v3.py",
    "simulation/carla_sensors_v3.py",
    "simulation/carla_environment_rgb_v3.py",
    "observation_builder_rgb_v3.py",
    "domain_randomization_runtime_v3.py",
    "train_ppo_rgb_v3.py",
]

# Intentionally NOT whole-file frozen:
# - action_controller_v3.py          : longitudinal may be revisited after REAL ax
# - domain_randomization_v3.py       : longitudinal nominal/ranges may be updated
PROVISIONAL_FILES = [
    "action_controller_v3.py",
    "domain_randomization_v3.py",
]


EXPECTED_FROZEN = {
    "timing": {
        "control_hz": 50.0,
        "control_dt_s": 0.020,
        "camera_fps": 30.0,
        "imu_hz": 50.0,
        "command_tx_hz": 50.0,
        "feedback_rx_hz": 50.0,
        "encoder_hz": 50.0,
        "servo_feedback_hz": 50.0,
        "extra_command_delay_nominal_ticks": 0,
        "extra_command_delay_range_ticks": [0, 1],
        "extra_command_delay_one_tick_probability": 0.25,
    },
    "lateral": {
        "steer_gain_positive": 1.1245,
        "steer_gain_negative": 1.4639,
        "dr_relative_half_range": 0.12,
        "status": "CALIBRATED",
    },
    "ppo_contract": {
        "obs_dim": 100,
        "vae_latent_dim": 95,
        "action_dim": 2,
        "action_semantics": [
            "steer_normalized_-1_to_1",
            "speed_cmd_real_equivalent_mps_0_to_1",
        ],
        "prev_command_semantics": "logical_applied_command",
    },
    "camera_control_behavior": {
        "camera_frame_reuse_between_50hz_ticks": True,
        "carla_imu_sensor_tick": 0.0,
    },
}


LONGITUDINAL_STATUS = {
    "status": "PROVISIONAL_PENDING_REAL_AX",
    "real_ax": "PENDING",
    "freeze_required_before_official_final_ppo": True,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def nearly_equal(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= float(tol)


def import_domain_randomization():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import domain_randomization_v3 as dr
    return dr


def current_semantic_snapshot():
    dr = import_domain_randomization()

    ranges = dr.DOMAIN_RANDOMIZATION_RANGES

    positive_nom = float(dr.STEER_GAIN_POSITIVE_NOMINAL)
    negative_nom = float(dr.STEER_GAIN_NEGATIVE_NOMINAL)

    pos_range = [
        float(ranges["steer_gain_positive"][0]),
        float(ranges["steer_gain_positive"][1]),
    ]
    neg_range = [
        float(ranges["steer_gain_negative"][0]),
        float(ranges["steer_gain_negative"][1]),
    ]

    pos_half = (
        (pos_range[1] - pos_range[0])
        / (2.0 * positive_nom)
    )
    neg_half = (
        (neg_range[1] - neg_range[0])
        / (2.0 * negative_nom)
    )

    return {
        "frozen": {
            "control_hz": float(dr.CONTROL_HZ),
            "control_dt_s": float(dr.CONTROL_DT_S),
            "camera_fps": float(dr.CAMERA_FPS),
            "steer_gain_positive": positive_nom,
            "steer_gain_negative": negative_nom,
            "steer_gain_positive_range": pos_range,
            "steer_gain_negative_range": neg_range,
            "steer_gain_positive_relative_half_range": float(pos_half),
            "steer_gain_negative_relative_half_range": float(neg_half),
            "use_gear_autobox": bool(dr.AUTOBOX_NOMINAL),
            "gear_switch_time_s": float(
                dr.GEAR_SWITCH_TIME_NOMINAL_S
            ),
            "extra_command_delay_range_ticks": [
                int(ranges["extra_command_delay_ticks"][0]),
                int(ranges["extra_command_delay_ticks"][1]),
            ],
            "extra_delay_one_tick_probability": float(
                dr.EXTRA_DELAY_ONE_TICK_PROBABILITY
            ),
        },
        "provisional_longitudinal": {
            "torque_scale_nominal": float(
                dr.TORQUE_SCALE_NOMINAL
            ),
            "max_brake_torque_nominal": float(
                dr.MAX_BRAKE_TORQUE_NOMINAL
            ),
            "torque_scale_range": [
                float(ranges["torque_scale"][0]),
                float(ranges["torque_scale"][1]),
            ],
            "max_brake_torque_range": [
                float(ranges["max_brake_torque"][0]),
                float(ranges["max_brake_torque"][1]),
            ],
        },
    }


def collect_file_hashes(relative_paths):
    result = {}

    for rel in relative_paths:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(
                "Thiếu file bắt buộc: {}".format(path)
            )
        result[rel] = sha256_file(path)

    return result


def semantic_errors(snapshot):
    errors = []
    frozen = snapshot["frozen"]

    checks = [
        (
            "CONTROL_HZ",
            frozen["control_hz"],
            EXPECTED_FROZEN["timing"]["control_hz"],
        ),
        (
            "CONTROL_DT_S",
            frozen["control_dt_s"],
            EXPECTED_FROZEN["timing"]["control_dt_s"],
        ),
        (
            "CAMERA_FPS",
            frozen["camera_fps"],
            EXPECTED_FROZEN["timing"]["camera_fps"],
        ),
        (
            "STEER_GAIN_POSITIVE",
            frozen["steer_gain_positive"],
            EXPECTED_FROZEN["lateral"]["steer_gain_positive"],
        ),
        (
            "STEER_GAIN_NEGATIVE",
            frozen["steer_gain_negative"],
            EXPECTED_FROZEN["lateral"]["steer_gain_negative"],
        ),
        (
            "STEER + DR half range",
            frozen[
                "steer_gain_positive_relative_half_range"
            ],
            EXPECTED_FROZEN["lateral"]["dr_relative_half_range"],
        ),
        (
            "STEER - DR half range",
            frozen[
                "steer_gain_negative_relative_half_range"
            ],
            EXPECTED_FROZEN["lateral"]["dr_relative_half_range"],
        ),
        (
            "GEAR_SWITCH_TIME",
            frozen["gear_switch_time_s"],
            0.0,
        ),
        (
            "P(delay=1)",
            frozen["extra_delay_one_tick_probability"],
            EXPECTED_FROZEN["timing"][
                "extra_command_delay_one_tick_probability"
            ],
        ),
    ]

    for name, actual, expected in checks:
        if not nearly_equal(actual, expected, 1e-8):
            errors.append(
                "{} changed: actual={} expected={}".format(
                    name,
                    actual,
                    expected,
                )
            )

    if frozen["use_gear_autobox"] is not True:
        errors.append(
            "AUTOBOX changed: actual={} expected=True".format(
                frozen["use_gear_autobox"]
            )
        )

    expected_delay_range = EXPECTED_FROZEN["timing"][
        "extra_command_delay_range_ticks"
    ]

    if (
        frozen["extra_command_delay_range_ticks"]
        != expected_delay_range
    ):
        errors.append(
            "Delay range changed: actual={} expected={}".format(
                frozen["extra_command_delay_range_ticks"],
                expected_delay_range,
            )
        )

    return errors


def build_manifest():
    snapshot = current_semantic_snapshot()
    errors = semantic_errors(snapshot)

    if errors:
        print("Không thể freeze vì semantic config không đúng:")
        for err in errors:
            print("  -", err)
        raise RuntimeError("GĐ7 semantic preflight FAIL.")

    manifest = {
        "schema": "gd7_freeze_manifest_v1",
        "created_local": datetime.now().isoformat(),
        "project_root": str(ROOT),

        "status": {
            "gd4_lateral_yaw": "PASS_FROZEN",
            "gd4_longitudinal": "PROVISIONAL_PENDING_REAL_AX",
            "gd5_timing_latency": "PASS_FROZEN",
            "gd6_domain_randomization": "PASS_FROZEN_ARCHITECTURE",
            "gd7": "PARTIAL_FREEZE_PENDING_REAL_AX",
        },

        "frozen_expected": EXPECTED_FROZEN,
        "longitudinal_status": LONGITUDINAL_STATUS,

        "frozen_file_hashes": collect_file_hashes(
            FROZEN_FILES
        ),
        "provisional_file_hashes_at_freeze_time": collect_file_hashes(
            PROVISIONAL_FILES
        ),

        "semantic_snapshot": snapshot,

        "notes": [
            "Do not edit frozen files without intentional GĐ7 re-freeze.",
            "action_controller_v3.py may be revisited only for final longitudinal calibration after REAL ax is fixed.",
            "domain_randomization_v3.py longitudinal nominal/ranges may be updated after REAL ax; frozen timing/lateral fields must remain unchanged.",
            "Official final PPO remains blocked until longitudinal status becomes FROZEN.",
        ],
    }

    return manifest


def freeze():
    if MANIFEST_PATH.exists():
        print(
            "Manifest đã tồn tại: {}".format(
                MANIFEST_PATH
            )
        )
        print(
            "Không overwrite tự động. Nếu muốn re-freeze có chủ đích, "
            "đổi tên/xóa manifest cũ rồi chạy --freeze."
        )
        return 2

    manifest = build_manifest()

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 96)
    print("GĐ7 PARTIAL FREEZE CREATED")
    print("=" * 96)
    print("MANIFEST :", MANIFEST_PATH)
    print("FROZEN   : timing / sensors / obs-action contract / lateral / DR architecture")
    print("PENDING  : REAL ax + final longitudinal")
    print("")
    print("IMPORTANT:")
    print("  Final official PPO is NOT unlocked yet.")
    print("  Sau khi ax REAL hoàn thiện: tune longitudinal -> update DR -> re-freeze GĐ7.")
    print("=" * 96)

    return 0


def verify():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Chưa có manifest. Chạy --freeze trước: {}".format(
                MANIFEST_PATH
            )
        )

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    failed = []
    warnings = []

    # 1) Whole-file frozen checks.
    frozen_hashes = manifest["frozen_file_hashes"]

    for rel, expected_hash in frozen_hashes.items():
        path = ROOT / rel

        if not path.is_file():
            failed.append(
                "Frozen file missing: {}".format(rel)
            )
            continue

        actual_hash = sha256_file(path)

        if actual_hash != expected_hash:
            failed.append(
                "Frozen file changed: {}".format(rel)
            )

    # 2) Semantic frozen checks.
    snapshot = current_semantic_snapshot()

    for err in semantic_errors(snapshot):
        failed.append(
            "Frozen semantic changed: {}".format(err)
        )

    # 3) Provisional longitudinal changes are warnings only.
    old_provisional = manifest[
        "semantic_snapshot"
    ]["provisional_longitudinal"]

    new_provisional = snapshot[
        "provisional_longitudinal"
    ]

    if old_provisional != new_provisional:
        warnings.append(
            "Longitudinal provisional config changed. "
            "This is allowed only for final REAL-ax calibration, "
            "but GĐ7 must be re-frozen afterwards."
        )

    old_provisional_hashes = manifest.get(
        "provisional_file_hashes_at_freeze_time",
        {},
    )

    for rel, old_hash in old_provisional_hashes.items():
        path = ROOT / rel

        if path.is_file():
            new_hash = sha256_file(path)

            if new_hash != old_hash:
                warnings.append(
                    "Provisional file changed: {}".format(rel)
                )

    print("=" * 96)
    print("GĐ7 FREEZE VERIFY")
    print("=" * 96)

    if failed:
        print("RESULT: FAIL")
        print("")
        for item in failed:
            print("  FAIL:", item)
    else:
        print("RESULT: PASS")
        print("  Frozen configuration has not drifted.")

    if warnings:
        print("")
        print("WARNINGS:")
        for item in warnings:
            print("  WARN:", item)

    print("")
    print("STATUS:")
    print("  Timing / latency          : FROZEN")
    print("  Lateral / yaw            : FROZEN")
    print("  DR architecture          : FROZEN")
    print("  Obs/action contract      : FROZEN")
    print("  Longitudinal / REAL ax   : PROVISIONAL / PENDING")
    print("=" * 96)

    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--freeze",
        action="store_true",
        help="Create GĐ7 freeze manifest.",
    )

    group.add_argument(
        "--verify",
        action="store_true",
        help="Verify current project against GĐ7 manifest.",
    )

    args = parser.parse_args()

    if args.freeze:
        code = freeze()
    else:
        code = verify()

    raise SystemExit(code)


if __name__ == "__main__":
    main()
