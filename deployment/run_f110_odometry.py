#!/usr/bin/env python3
"""Run F1/10 wheel+MPU6050 odometry without ROS.

Encoder transport protocol
--------------------------
An Arduino/ESP32/other counter MCU sends one newline-terminated JSON object:

    {"front_left":123,"front_right":124,"rear_left":120,"rear_right":121}

Counts must be sampled together and be cumulative.  Default serial speed is
921600 baud.  This process reads MPU6050 locally over I2C, estimates odometry,
prints JSON Lines to stdout, and optionally records a CSV file.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Dict

from f110_wheel_imu_odometry import (
    F110WheelImuOdometry,
    ImuSample,
    MPU6050,
    OdometryConfig,
    WHEELS,
)


class SerialJsonEncoders:
    def __init__(self, port: str, baudrate: int, timeout_s: float) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Install pyserial: pip install pyserial") from exc
        self.serial = serial.Serial(port, baudrate, timeout=timeout_s)
        self.serial.reset_input_buffer()

    def read(self) -> Dict[str, int]:
        line = self.serial.readline()
        if not line:
            raise TimeoutError("Encoder serial timeout")
        try:
            payload = json.loads(line.decode("utf-8"))
            return {wheel: int(payload[wheel]) for wheel in WHEELS}
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid encoder packet: {line!r}") from exc


def load_config(path: Path) -> OdometryConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(OdometryConfig)}
    unknown = set(payload).difference(valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return OdometryConfig(**payload)


def calibrate_imu(estimator, imu, samples: int, sample_period_s: float) -> float:
    gyro_samples = []
    accel_norms = []
    print(
        f"Keep vehicle motionless: calibrating MPU6050 with {samples} samples...",
        file=sys.stderr,
    )
    for _ in range(samples):
        sample = imu.read()
        gyro_samples.append(sample.gyro_z_rad_s)
        accel_norms.append(
            math.sqrt(
                sample.accel_x_m_s2**2
                + sample.accel_y_m_s2**2
                + sample.accel_z_m_s2**2
            )
        )
        time.sleep(sample_period_s)

    if max(accel_norms) - min(accel_norms) > 0.8:
        raise RuntimeError("Vehicle moved/vibrated too much during IMU calibration")
    return estimator.calibrate_stationary_gyro(gyro_samples)


def open_csv(path: str):
    if not path:
        return None, None
    handle = open(path, "w", newline="", encoding="utf-8")
    names = [field.name for field in fields(type_placeholder_state())]
    writer = csv.DictWriter(handle, fieldnames=names)
    writer.writeheader()
    return handle, writer


def type_placeholder_state():
    # Imported lazily this way keeps the CSV schema tied to OdometryState.
    from f110_wheel_imu_odometry import OdometryState
    return OdometryState(0, 0, 0, 0, 0, 0, 0, False, False, 0, 0, 0, 0, 0, 0)


def run_hardware(args) -> int:
    config = load_config(Path(args.config))
    estimator = F110WheelImuOdometry(config)
    imu = MPU6050(bus_number=args.i2c_bus, address=int(args.i2c_address, 0))
    encoders = SerialJsonEncoders(args.encoder_port, args.baudrate, args.timeout)

    bias = calibrate_imu(estimator, imu, args.calibration_samples, 0.005)
    print(f"MPU6050 gyro-Z bias={bias:+.7f} rad/s", file=sys.stderr)

    csv_handle, csv_writer = open_csv(args.csv)
    bad_packets = 0
    try:
        while True:
            try:
                counts = encoders.read()
                sample = imu.read()
                timestamp = time.monotonic()
                state = estimator.update(timestamp, counts, sample)
                bad_packets = 0
            except (TimeoutError, ValueError, OSError) as exc:
                bad_packets += 1
                print(f"sensor warning ({bad_packets}): {exc}", file=sys.stderr)
                if bad_packets >= args.max_bad_packets:
                    raise RuntimeError("Too many consecutive sensor failures") from exc
                continue

            if state is None:
                continue
            row = asdict(state)
            print(json.dumps(row, separators=(",", ":")), flush=True)
            if csv_writer is not None:
                csv_writer.writerow(row)
                csv_handle.flush()
    except KeyboardInterrupt:
        return 0
    finally:
        if csv_handle is not None:
            csv_handle.close()


def run_self_test() -> int:
    config = OdometryConfig(
        wheel_radius_m=0.05,
        rear_track_width_m=0.29,
        ticks_per_revolution=2048,
        wheel_speed_filter_alpha=1.0,
        wheel_speed_median_window=1,
        left_right_accel_disagreement_m_s2=100.0,
    )
    estimator = F110WheelImuOdometry(config)
    estimator.calibrate_stationary_gyro(
        [0.02 + random.gauss(0.0, 0.002) for _ in range(300)]
    )
    counts = {wheel: 0 for wheel in WHEELS}
    timestamp = 0.0
    state = None

    # Two metres straight at 1 m/s.
    for _ in range(201):
        state = estimator.update(
            timestamp, counts, ImuSample(0.02, 0.0, 0.0, 9.80665)
        )
        timestamp += 0.01
        ticks = round(1.0 * 0.01 / (2.0 * math.pi * 0.05) * 2048)
        for wheel in WHEELS:
            counts[wheel] += ticks
    if state is None or abs(state.x_m - 2.0) > 0.08 or abs(state.y_m) > 0.03:
        raise AssertionError(f"Straight test failed: {state}")

    # Then turn for two seconds at 1 m/s and 0.5 rad/s.
    for _ in range(200):
        state = estimator.update(
            timestamp, counts, ImuSample(0.52, 0.0, 0.0, 9.80665)
        )
        timestamp += 0.01
        left = 1.0 - 0.5 * 0.29 / 2.0
        right = 1.0 + 0.5 * 0.29 / 2.0
        dl = round(left * 0.01 / (2.0 * math.pi * 0.05) * 2048)
        dr = round(right * 0.01 / (2.0 * math.pi * 0.05) * 2048)
        counts["front_left"] += dl
        counts["rear_left"] += dl
        counts["front_right"] += dr
        counts["rear_right"] += dr
    if state is None or not 0.85 < state.yaw_rad < 1.15:
        raise AssertionError(f"Turn test failed: {state}")
    print(json.dumps(asdict(state), indent=2))
    print("SELF-TEST PASSED", file=sys.stderr)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("f110_odometry_config.json")),
    )
    parser.add_argument("--encoder-port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=0.1)
    parser.add_argument("--max-bad-packets", type=int, default=20)
    parser.add_argument("--i2c-bus", type=int, default=1)
    parser.add_argument("--i2c-address", default="0x68")
    parser.add_argument("--calibration-samples", type=int, default=500)
    parser.add_argument("--csv", default="")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run_self_test() if arguments.self_test else run_hardware(arguments))
