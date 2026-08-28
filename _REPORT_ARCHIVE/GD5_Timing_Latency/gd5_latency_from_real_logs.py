# -*- coding: utf-8 -*-
"""
GĐ5 - STEP 1: REAL observed latency baseline from existing logs

Không cần ax hoàn thiện.
Không sửa PPO / CARLA physics.

Script tự đọc:
  Log/test2_acceleration_*.csv
  Log/test4_steering_yaw_*.csv

Mục tiêu:
1) Xác nhận chu kỳ REAL ~20 ms / 50 Hz.
2) SPEED path:
   speed_cmd -> motor_pwm
   speed_cmd -> encoder first pulse
   speed_cmd -> measurable speed
3) STEERING path:
   steer_cmd -> servo_target
   steer_cmd -> servo feedback movement
   steer_cmd -> 10% servo response
   steer_cmd -> observable yaw/gyro response

LƯU Ý:
- Đây là "observed end-to-end latency" từ log Jetson, KHÔNG phải latency
  thuần TX UART hay STM32 processing.
- Độ phân giải log hiện tại ~20 ms, nên latency dưới 20 ms chỉ có thể
  kết luận là "<= 1 control tick".
- gyro/yaw onset chứa cả servo + cơ học xe, không được coi là UART latency.

Chạy:
    python .\dynamics_calibration\analysis\gd5_latency_from_real_logs.py
"""

from __future__ import print_function

import argparse
import csv
import glob
import math
import re
import statistics
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
LOG_DIR = PROJECT_ROOT / "Log"


def finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except Exception:
        return default


def i(row, key, default=0):
    try:
        return int(float(row[key]))
    except Exception:
        return default


def read_csv(path):
    with open(str(path), "r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def median(values):
    values = [float(x) for x in values if finite(x)]
    if not values:
        return float("nan")
    return float(statistics.median(values))


def mean(values):
    values = [float(x) for x in values if finite(x)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def stdev_population(values):
    values = [float(x) for x in values if finite(x)]
    if not values:
        return float("nan")
    m = mean(values)
    return float(math.sqrt(sum((x - m) ** 2 for x in values) / len(values)))


def pct(values, p):
    values = sorted(float(x) for x in values if finite(x))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * float(p)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def dt_ms_between_rows(a, b):
    return f(b, "time_ms") - f(a, "time_ms")


def first_index(rows, predicate, start=0):
    for idx in range(max(0, int(start)), len(rows)):
        try:
            if predicate(rows[idx]):
                return idx
        except Exception:
            pass
    return None


def group_by_segment(rows):
    out = {}
    for row in rows:
        out.setdefault(row.get("segment", ""), []).append(row)
    return out


def sample_period_stats(paths):
    diffs = []
    per_file = []

    for path in paths:
        rows = read_csv(path)
        local = []

        prev = None
        for row in rows:
            t = f(row, "time_ms")
            if not finite(t):
                continue
            if prev is not None:
                d = t - prev
                if 0.0 < d < 200.0:
                    diffs.append(d)
                    local.append(d)
            prev = t

        if local:
            per_file.append({
                "file": str(path),
                "median_ms": median(local),
                "p05_ms": pct(local, 0.05),
                "p95_ms": pct(local, 0.95),
                "std_ms": stdev_population(local),
            })

    return diffs, per_file


def analyze_speed_file(path):
    rows = read_csv(path)
    groups = group_by_segment(rows)
    results = []

    accel_names = [
        name for name in groups
        if name.startswith("accel_0_to_")
    ]

    for name in sorted(accel_names):
        seg = groups[name]
        if len(seg) < 2:
            continue

        # Segment typically starts with one 0-command row, then command step.
        cmd_idx = first_index(
            seg,
            lambda r: abs(f(r, "speed_cmd_mps")) > 1e-9,
        )
        if cmd_idx is None:
            continue

        target = f(seg[cmd_idx], "speed_cmd_mps")
        t0 = f(seg[cmd_idx], "time_ms")

        # Baseline from rows before step if available.
        base_pwm = mean(
            [f(r, "motor_pwm") for r in seg[:cmd_idx]]
        )
        if not finite(base_pwm):
            base_pwm = 0.0

        base_speed = mean(
            [f(r, "speed_mps") for r in seg[:cmd_idx]]
        )
        if not finite(base_speed):
            base_speed = 0.0

        pwm_idx = first_index(
            seg,
            lambda r: abs(f(r, "motor_pwm") - base_pwm) >= 1.0,
            start=cmd_idx,
        )

        enc_idx = first_index(
            seg,
            lambda r: abs(i(r, "encoder_delta_count")) >= 1,
            start=cmd_idx,
        )

        # Measurable speed threshold: 0.01 m/s.
        speed_idx = first_index(
            seg,
            lambda r: abs(f(r, "speed_mps") - base_speed) >= 0.01,
            start=cmd_idx,
        )

        # 10% physical response, consistent with step-response notion.
        t10_idx = first_index(
            seg,
            lambda r: f(r, "speed_mps") >= 0.10 * target,
            start=cmd_idx,
        )

        def latency(idx):
            if idx is None:
                return float("nan")
            return f(seg[idx], "time_ms") - t0

        results.append({
            "case": name,
            "target_mps": target,
            "cmd_time_ms": t0,
            "cmd_to_motor_pwm_ms": latency(pwm_idx),
            "cmd_to_encoder_pulse_ms": latency(enc_idx),
            "cmd_to_speed_0p01_ms": latency(speed_idx),
            "cmd_to_speed_10pct_ms": latency(t10_idx),
        })

    return results


def analyze_steer_file(path):
    rows = read_csv(path)
    groups = group_by_segment(rows)
    results = []

    yaw_names = [
        name for name in groups
        if name.startswith("yaw_")
    ]

    for name in sorted(yaw_names):
        seg = groups[name]
        if len(seg) < 3:
            continue

        cmd_idx = first_index(
            seg,
            lambda r: abs(f(r, "steer_cmd_deg")) >= 1.0,
        )
        if cmd_idx is None:
            continue

        cmd = f(seg[cmd_idx], "steer_cmd_deg")
        direction = 1.0 if cmd > 0.0 else -1.0
        t0 = f(seg[cmd_idx], "time_ms")

        # Normally row 0 is center state just before step.
        baseline_rows = seg[:cmd_idx]
        if not baseline_rows:
            baseline_rows = seg[:1]

        base_target = mean(
            [f(r, "servo_target_raw") for r in baseline_rows]
        )
        base_fb_deg = mean(
            [f(r, "steer_fb_deg") for r in baseline_rows]
        )
        base_gyro = mean(
            [f(r, "gyro_z_rad_s") for r in baseline_rows]
        )

        target_idx = first_index(
            seg,
            lambda r: abs(
                f(r, "servo_target_raw") - base_target
            ) >= 1.0,
            start=cmd_idx,
        )

        # Any meaningful servo feedback movement.
        fb_move_idx = first_index(
            seg,
            lambda r: direction * (
                f(r, "steer_fb_deg") - base_fb_deg
            ) >= 0.5,
            start=cmd_idx,
        )

        # 10% of 12 deg = 1.2 deg.
        fb10_idx = first_index(
            seg,
            lambda r: direction * (
                f(r, "steer_fb_deg") - base_fb_deg
            ) >= 0.10 * abs(cmd),
            start=cmd_idx,
        )

        # Observable gyro onset.
        # Threshold 0.05 rad/s in commanded direction relative to the
        # pre-step sample. This includes mechanical response.
        gyro_idx = first_index(
            seg,
            lambda r: direction * (
                f(r, "gyro_z_rad_s") - base_gyro
            ) >= 0.05,
            start=cmd_idx,
        )

        def latency(idx):
            if idx is None:
                return float("nan")
            return f(seg[idx], "time_ms") - t0

        results.append({
            "case": name,
            "steer_cmd_deg": cmd,
            "cmd_time_ms": t0,
            "cmd_to_servo_target_ms": latency(target_idx),
            "cmd_to_servo_fb_move_ms": latency(fb_move_idx),
            "cmd_to_servo_fb_10pct_ms": latency(fb10_idx),
            "cmd_to_gyro_observed_ms": latency(gyro_idx),
        })

    return results


def print_latency_table(title, rows, columns):
    print("")
    print("=" * 112)
    print(title)
    print("=" * 112)

    header = " | ".join("{:>20}".format(c) for c in columns)
    print(header)
    print("-" * 112)

    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c)
            if isinstance(v, float):
                if math.isfinite(v):
                    vals.append("{:>20.1f}".format(v))
                else:
                    vals.append("{:>20}".format("nan"))
            else:
                vals.append("{:>20}".format(str(v)))
        print(" | ".join(vals))

    print("-" * 112)


def aggregate(rows, keys):
    out = {}
    for key in keys:
        vals = [r.get(key) for r in rows]
        vals = [float(v) for v in vals if finite(v)]
        if vals:
            out[key] = {
                "median": median(vals),
                "min": min(vals),
                "max": max(vals),
            }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(LOG_DIR),
        help="REAL Log directory. Default: project/Log",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)

    accel_name_re = re.compile(
        r"^test2_acceleration_\d{8}_\d{6}\.csv$"
    )
    steer_name_re = re.compile(
        r"^test4_steering_yaw_\d{8}_\d{6}\.csv$"
    )

    accel_paths = sorted(
        p for p in log_dir.glob("test2_acceleration_*.csv")
        if accel_name_re.match(p.name)
    )

    steer_paths = sorted(
        p for p in log_dir.glob("test4_steering_yaw_*.csv")
        if steer_name_re.match(p.name)
    )

    # Prefer newest REAL acceleration raw file if duplicates exist.
    if accel_paths:
        accel_paths = [accel_paths[-1]]

    all_paths = accel_paths + steer_paths

    if not all_paths:
        raise RuntimeError(
            "Không tìm thấy REAL test2/test4 raw CSV trong {}".format(
                log_dir
            )
        )

    print("=" * 112)
    print("GĐ5 - STEP 1 | REAL OBSERVED LATENCY BASELINE")
    print("=" * 112)
    print("LOG DIR :", log_dir)
    print("ACCEL   :", [p.name for p in accel_paths])
    print("STEER   :", [p.name for p in steer_paths])
    print("=" * 112)

    diffs, per_file = sample_period_stats(all_paths)

    print("")
    print("REAL LOOP / LOG PERIOD")
    print(
        "median={:.2f} ms | mean={:.2f} ms | std={:.2f} ms | "
        "p05={:.2f} | p95={:.2f} | N={}".format(
            median(diffs),
            mean(diffs),
            stdev_population(diffs),
            pct(diffs, 0.05),
            pct(diffs, 0.95),
            len(diffs),
        )
    )

    speed_rows = []
    for path in accel_paths:
        speed_rows.extend(analyze_speed_file(path))

    steer_rows = []
    for path in steer_paths:
        steer_rows.extend(analyze_steer_file(path))

    print_latency_table(
        "SPEED COMMAND -> OBSERVED RESPONSE",
        speed_rows,
        [
            "target_mps",
            "cmd_to_motor_pwm_ms",
            "cmd_to_encoder_pulse_ms",
            "cmd_to_speed_0p01_ms",
            "cmd_to_speed_10pct_ms",
        ],
    )

    print_latency_table(
        "STEER COMMAND -> OBSERVED RESPONSE",
        steer_rows,
        [
            "steer_cmd_deg",
            "cmd_to_servo_target_ms",
            "cmd_to_servo_fb_move_ms",
            "cmd_to_servo_fb_10pct_ms",
            "cmd_to_gyro_observed_ms",
        ],
    )

    speed_ag = aggregate(
        speed_rows,
        [
            "cmd_to_motor_pwm_ms",
            "cmd_to_encoder_pulse_ms",
            "cmd_to_speed_0p01_ms",
            "cmd_to_speed_10pct_ms",
        ],
    )

    steer_ag = aggregate(
        steer_rows,
        [
            "cmd_to_servo_target_ms",
            "cmd_to_servo_fb_move_ms",
            "cmd_to_servo_fb_10pct_ms",
            "cmd_to_gyro_observed_ms",
        ],
    )

    print("")
    print("=" * 112)
    print("GĐ5 STEP 1 - INTERPRETATION")
    print("=" * 112)

    period = median(diffs)
    print(
        "1) REAL loop/log period ~= {:.1f} ms ({:.1f} Hz).".format(
            period,
            1000.0 / period if finite(period) and period > 0 else float("nan"),
        )
    )

    if "cmd_to_motor_pwm_ms" in speed_ag:
        x = speed_ag["cmd_to_motor_pwm_ms"]["median"]
        print(
            "2) speed_cmd -> motor_pwm observed median = {:.1f} ms.".format(x)
        )

    if "cmd_to_encoder_pulse_ms" in speed_ag:
        x = speed_ag["cmd_to_encoder_pulse_ms"]["median"]
        print(
            "3) speed_cmd -> first encoder pulse observed median = {:.1f} ms.".format(x)
        )

    if "cmd_to_servo_target_ms" in steer_ag:
        x = steer_ag["cmd_to_servo_target_ms"]["median"]
        print(
            "4) steer_cmd -> servo target observed median = {:.1f} ms.".format(x)
        )

    if "cmd_to_servo_fb_move_ms" in steer_ag:
        x = steer_ag["cmd_to_servo_fb_move_ms"]["median"]
        print(
            "5) steer_cmd -> first servo feedback movement median = {:.1f} ms.".format(x)
        )

    if "cmd_to_gyro_observed_ms" in steer_ag:
        x = steer_ag["cmd_to_gyro_observed_ms"]["median"]
        print(
            "6) steer_cmd -> observable gyro/yaw response median = {:.1f} ms "
            "(includes servo + vehicle mechanics).".format(x)
        )

    print("")
    print(
        "IMPORTANT: với log 50 Hz, mọi kết quả 0-20 ms chỉ cho biết "
        "'trong <= 1 tick'. Muốn tách chính xác Jetson TX -> STM32 RX/actuate "
        "phải thêm timestamp/sequence ID ở firmware rồi đo ở GĐ5 Step 2."
    )
    print("")
    print(
        "NEXT: gửi toàn bộ terminal output này. Sau đó quyết định có cần "
        "GĐ5 Step 2 firmware timestamp hay đã đủ để đặt nominal latency trong SIM."
    )


if __name__ == "__main__":
    main()
