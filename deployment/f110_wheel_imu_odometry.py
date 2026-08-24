#!/usr/bin/env python3
"""Wheel encoder + MPU6050 odometry for a planar F1/10 Ackermann car.

The estimator deliberately does not double-integrate accelerometer data.  Wheel
encoders provide longitudinal distance, while a small Kalman filter combines
rear-wheel differential yaw rate with the MPU6050 Z gyroscope and estimates
gyro bias online.

Coordinate convention:
    x forward, y left, z up; positive yaw is counter-clockwise.

Expected units:
    timestamp_s: seconds from a monotonic clock
    encoder counts: integer ticks (signed, increasing while driving forward)
    gyro_z_rad_s: rad/s
    accelerometer: m/s^2
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import time
from typing import Deque, Dict, Mapping, Optional


WHEELS = ("front_left", "front_right", "rear_left", "rear_right")


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class ImuSample:
    gyro_z_rad_s: float
    accel_x_m_s2: float
    accel_y_m_s2: float
    accel_z_m_s2: float


@dataclass(frozen=True)
class OdometryState:
    timestamp_s: float
    x_m: float
    y_m: float
    yaw_rad: float
    speed_m_s: float
    yaw_rate_rad_s: float
    gyro_bias_rad_s: float
    stationary: bool
    slip_detected: bool
    position_variance_m2: float
    yaw_variance_rad2: float
    front_left_speed_m_s: float
    front_right_speed_m_s: float
    rear_left_speed_m_s: float
    rear_right_speed_m_s: float


@dataclass
class OdometryConfig:
    wheel_radius_m: float = 0.050
    rear_track_width_m: float = 0.290
    ticks_per_revolution: int = 2048
    encoder_counter_modulus: Optional[int] = None

    # Set a sign to -1 when that encoder decreases while the car moves forward.
    encoder_signs: Mapping[str, float] = None

    min_dt_s: float = 0.002
    max_dt_s: float = 0.200
    max_wheel_speed_m_s: float = 8.0
    wheel_speed_filter_alpha: float = 0.35
    wheel_speed_median_window: int = 3

    # Stationary detection. MPU6050 acceleration includes gravity.
    stationary_speed_m_s: float = 0.025
    stationary_gyro_rad_s: float = 0.025
    stationary_accel_tolerance_m_s2: float = 0.35
    stationary_required_samples: int = 15

    # Slip/fault detection from disagreement between front/rear axles.
    axle_speed_disagreement_m_s: float = 0.35
    left_right_accel_disagreement_m_s2: float = 4.0

    # Two-state Kalman filter: [true yaw rate, gyro bias].
    yaw_acceleration_process_std: float = 1.5
    gyro_bias_random_walk_std: float = 0.003
    gyro_measurement_std: float = 0.025
    wheel_yaw_measurement_std: float = 0.12
    wheel_yaw_slip_measurement_std: float = 1.5

    initial_gyro_bias_std: float = 0.10
    initial_yaw_rate_std: float = 0.30

    # Covariance growth for the integrated pose.
    distance_noise_fraction: float = 0.02
    distance_noise_floor_m: float = 0.001

    def __post_init__(self) -> None:
        if self.encoder_signs is None:
            self.encoder_signs = {wheel: 1.0 for wheel in WHEELS}
        missing = set(WHEELS).difference(self.encoder_signs)
        if missing:
            raise ValueError(f"Missing encoder signs for: {sorted(missing)}")
        if self.wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be positive")
        if self.rear_track_width_m <= 0.0:
            raise ValueError("rear_track_width_m must be positive")
        if self.ticks_per_revolution <= 0:
            raise ValueError("ticks_per_revolution must be positive")
        if (
            self.encoder_counter_modulus is not None
            and self.encoder_counter_modulus <= 1
        ):
            raise ValueError("encoder_counter_modulus must be > 1 or None")


class ScalarMedianEma:
    """Short median filter followed by an exponential moving average."""

    def __init__(self, window: int, alpha: float) -> None:
        self.samples: Deque[float] = deque(maxlen=max(1, int(window)))
        self.alpha = clamp(float(alpha), 0.0, 1.0)
        self.value: Optional[float] = None

    def update(self, sample: float) -> float:
        self.samples.append(float(sample))
        median = float(statistics.median(self.samples))
        if self.value is None:
            self.value = median
        else:
            self.value += self.alpha * (median - self.value)
        return self.value


class YawRateBiasKalman:
    """Estimate true yaw rate and MPU6050 gyro-Z bias.

    State x = [omega, bias].  Measurements are:
        gyro_z = omega + bias + noise
        wheel_yaw_rate = omega + noise

    Wheel yaw observations make gyro bias observable without a magnetometer.
    """

    def __init__(self, config: OdometryConfig) -> None:
        self.config = config
        self.omega = 0.0
        self.gyro_bias = 0.0
        self.p00 = config.initial_yaw_rate_std**2
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = config.initial_gyro_bias_std**2

    @property
    def yaw_rate(self) -> float:
        return self.omega

    @property
    def bias(self) -> float:
        return self.gyro_bias

    @property
    def yaw_rate_variance(self) -> float:
        return self.p00

    def set_bias(self, bias_rad_s: float, variance: float = 1e-5) -> None:
        self.gyro_bias = float(bias_rad_s)
        self.p11 = max(float(variance), 1e-10)

    def predict(self, dt: float) -> None:
        # Random-walk yaw rate and slowly varying gyro bias.
        q_omega = (self.config.yaw_acceleration_process_std * dt) ** 2
        q_bias = (self.config.gyro_bias_random_walk_std**2) * dt
        self.p00 += q_omega
        self.p11 += q_bias

    def _update(
        self,
        measurement: float,
        h0: float,
        h1: float,
        variance: float,
    ) -> None:
        innovation = float(measurement) - (
            h0 * self.omega + h1 * self.gyro_bias
        )
        ph0 = self.p00 * h0 + self.p01 * h1
        ph1 = self.p10 * h0 + self.p11 * h1
        S = h0 * ph0 + h1 * ph1 + variance

        # Reject gross sensor spikes using a 5-sigma innovation gate.
        if innovation**2 > 25.0 * max(S, 1e-12):
            return

        k0 = ph0 / S
        k1 = ph1 / S
        self.omega += k0 * innovation
        self.gyro_bias += k1 * innovation

        # Scalar Kalman covariance update, followed by symmetrization.
        old00, old01 = self.p00, self.p01
        old10, old11 = self.p10, self.p11
        self.p00 = old00 - k0 * (h0 * old00 + h1 * old10)
        self.p01 = old01 - k0 * (h0 * old01 + h1 * old11)
        self.p10 = old10 - k1 * (h0 * old00 + h1 * old10)
        self.p11 = old11 - k1 * (h0 * old01 + h1 * old11)
        off_diagonal = 0.5 * (self.p01 + self.p10)
        self.p01 = off_diagonal
        self.p10 = off_diagonal
        self.p00 = max(self.p00, 1e-12)
        self.p11 = max(self.p11, 1e-12)

    def update_gyro(self, gyro_z_rad_s: float) -> None:
        self._update(
            gyro_z_rad_s,
            1.0,
            1.0,
            self.config.gyro_measurement_std**2,
        )

    def update_wheels(self, wheel_yaw_rate: float, slipping: bool) -> None:
        std = (
            self.config.wheel_yaw_slip_measurement_std
            if slipping
            else self.config.wheel_yaw_measurement_std
        )
        self._update(wheel_yaw_rate, 1.0, 0.0, std**2)

    def enforce_stationary(self, gyro_z_rad_s: float) -> None:
        # At rest omega=0, so gyro directly observes bias.
        self._update(
            0.0,
            1.0,
            0.0,
            1e-6,
        )
        self._update(
            gyro_z_rad_s,
            0.0,
            1.0,
            self.config.gyro_measurement_std**2,
        )


class F110WheelImuOdometry:
    """Sensor-agnostic odometry estimator.

    Call update() whenever a synchronized encoder/IMU sample is available.
    Use time.monotonic() for timestamp_s; never use wall-clock time.
    """

    def __init__(self, config: Optional[OdometryConfig] = None) -> None:
        self.config = config or OdometryConfig()
        self.metres_per_tick = (
            2.0 * math.pi * self.config.wheel_radius_m
            / float(self.config.ticks_per_revolution)
        )

        self.filters = {
            wheel: ScalarMedianEma(
                self.config.wheel_speed_median_window,
                self.config.wheel_speed_filter_alpha,
            )
            for wheel in WHEELS
        }
        self.yaw_filter = YawRateBiasKalman(self.config)

        self.previous_timestamp_s: Optional[float] = None
        self.previous_counts: Optional[Dict[str, int]] = None
        self.previous_speeds = {wheel: 0.0 for wheel in WHEELS}

        self.x_m = 0.0
        self.y_m = 0.0
        self.yaw_rad = 0.0
        self.position_variance_m2 = 1e-6
        self.yaw_variance_rad2 = 1e-6
        self.stationary_counter = 0

    def reset(self, x_m: float = 0.0, y_m: float = 0.0,
              yaw_rad: float = 0.0) -> None:
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.yaw_rad = wrap_angle(float(yaw_rad))
        self.position_variance_m2 = 1e-6
        self.yaw_variance_rad2 = 1e-6
        self.previous_timestamp_s = None
        self.previous_counts = None
        self.stationary_counter = 0

    def calibrate_stationary_gyro(
        self,
        gyro_samples_rad_s,
    ) -> float:
        """Robust initial bias calibration while the vehicle is motionless."""
        values = [float(value) for value in gyro_samples_rad_s]
        values = [value for value in values if math.isfinite(value)]
        if len(values) < 50:
            raise ValueError("At least 50 stationary gyro samples are required")

        median = float(statistics.median(values))
        mad = float(statistics.median(abs(value - median) for value in values))
        robust_sigma = max(1.4826 * mad, 1e-6)
        inliers = [
            value
            for value in values
            if abs(value - median) <= 4.0 * robust_sigma
        ]
        if len(inliers) < max(20, len(values) // 2):
            raise ValueError("Too many gyro outliers during calibration")

        bias = float(statistics.fmean(inliers))
        variance = max(float(statistics.pvariance(inliers)), 1e-8)
        self.yaw_filter.set_bias(bias, variance)
        return bias

    def update(
        self,
        timestamp_s: float,
        encoder_counts: Mapping[str, int],
        imu: ImuSample,
    ) -> Optional[OdometryState]:
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not all(wheel in encoder_counts for wheel in WHEELS):
            raise ValueError(f"encoder_counts must contain {WHEELS}")
        if not all(
            math.isfinite(value)
            for value in (
                imu.gyro_z_rad_s,
                imu.accel_x_m_s2,
                imu.accel_y_m_s2,
                imu.accel_z_m_s2,
            )
        ):
            raise ValueError("IMU sample contains NaN or infinity")

        counts = {wheel: int(encoder_counts[wheel]) for wheel in WHEELS}

        if self.previous_timestamp_s is None or self.previous_counts is None:
            self.previous_timestamp_s = float(timestamp_s)
            self.previous_counts = counts
            return None

        dt = float(timestamp_s) - self.previous_timestamp_s
        if dt < self.config.min_dt_s:
            return None
        if dt > self.config.max_dt_s:
            # Avoid integrating through a sensor dropout.
            self.previous_timestamp_s = float(timestamp_s)
            self.previous_counts = counts
            self.stationary_counter = 0
            return None

        self.previous_timestamp_s = float(timestamp_s)

        raw_speeds: Dict[str, float] = {}
        for wheel in WHEELS:
            delta_ticks = counts[wheel] - self.previous_counts[wheel]
            modulus = self.config.encoder_counter_modulus
            if modulus is not None:
                half = modulus // 2
                delta_ticks = (delta_ticks + half) % modulus - half
            signed_delta = delta_ticks * float(self.config.encoder_signs[wheel])
            raw_speed = signed_delta * self.metres_per_tick / dt
            raw_speeds[wheel] = clamp(
                raw_speed,
                -self.config.max_wheel_speed_m_s,
                self.config.max_wheel_speed_m_s,
            )

        self.previous_counts = counts
        speeds = {
            wheel: self.filters[wheel].update(raw_speeds[wheel])
            for wheel in WHEELS
        }

        rear_speed = 0.5 * (speeds["rear_left"] + speeds["rear_right"])
        front_speed = 0.5 * (speeds["front_left"] + speeds["front_right"])
        wheel_yaw_rate = (
            speeds["rear_right"] - speeds["rear_left"]
        ) / self.config.rear_track_width_m

        accel_norm = math.sqrt(
            imu.accel_x_m_s2**2
            + imu.accel_y_m_s2**2
            + imu.accel_z_m_s2**2
        )
        accel_near_gravity = (
            abs(accel_norm - 9.80665)
            < self.config.stationary_accel_tolerance_m_s2
        )
        stationary_candidate = (
            abs(rear_speed) < self.config.stationary_speed_m_s
            and abs(imu.gyro_z_rad_s - self.yaw_filter.bias)
            < self.config.stationary_gyro_rad_s
            and accel_near_gravity
        )
        self.stationary_counter = (
            self.stationary_counter + 1 if stationary_candidate else 0
        )
        stationary = (
            self.stationary_counter
            >= self.config.stationary_required_samples
        )

        axle_disagreement = abs(front_speed - rear_speed)
        wheel_accelerations = [
            (speeds[wheel] - self.previous_speeds[wheel]) / dt
            for wheel in WHEELS
        ]
        wheel_accel_disagreement = (
            max(wheel_accelerations) - min(wheel_accelerations)
        )
        slip_detected = (
            axle_disagreement
            > self.config.axle_speed_disagreement_m_s
            or wheel_accel_disagreement
            > self.config.left_right_accel_disagreement_m_s2
        )
        self.previous_speeds = speeds

        self.yaw_filter.predict(dt)
        self.yaw_filter.update_gyro(float(imu.gyro_z_rad_s))
        self.yaw_filter.update_wheels(wheel_yaw_rate, slip_detected)

        if stationary:
            self.yaw_filter.enforce_stationary(float(imu.gyro_z_rad_s))
            speed = 0.0
            yaw_rate = 0.0
        else:
            speed = rear_speed
            yaw_rate = self.yaw_filter.yaw_rate

        delta_yaw = yaw_rate * dt
        yaw_mid = self.yaw_rad + 0.5 * delta_yaw
        distance = speed * dt

        self.x_m += distance * math.cos(yaw_mid)
        self.y_m += distance * math.sin(yaw_mid)
        self.yaw_rad = wrap_angle(self.yaw_rad + delta_yaw)

        distance_std = (
            self.config.distance_noise_floor_m
            + self.config.distance_noise_fraction * abs(distance)
        )
        if slip_detected:
            distance_std *= 5.0
        self.position_variance_m2 += distance_std**2
        self.yaw_variance_rad2 += max(
            self.yaw_filter.yaw_rate_variance * dt * dt,
            1e-9,
        )

        return OdometryState(
            timestamp_s=float(timestamp_s),
            x_m=self.x_m,
            y_m=self.y_m,
            yaw_rad=self.yaw_rad,
            speed_m_s=speed,
            yaw_rate_rad_s=yaw_rate,
            gyro_bias_rad_s=self.yaw_filter.bias,
            stationary=stationary,
            slip_detected=slip_detected,
            position_variance_m2=self.position_variance_m2,
            yaw_variance_rad2=self.yaw_variance_rad2,
            front_left_speed_m_s=speeds["front_left"],
            front_right_speed_m_s=speeds["front_right"],
            rear_left_speed_m_s=speeds["rear_left"],
            rear_right_speed_m_s=speeds["rear_right"],
        )


class MPU6050:
    """Minimal MPU6050 I2C reader using smbus2.

    Install with: pip install smbus2
    The default full-scale settings are +/-2 g and +/-250 deg/s.
    """

    PWR_MGMT_1 = 0x6B
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B

    def __init__(self, bus_number: int = 1, address: int = 0x68) -> None:
        from smbus2 import SMBus

        self.bus = SMBus(bus_number)
        self.address = address
        self.bus.write_byte_data(address, self.PWR_MGMT_1, 0x00)
        time.sleep(0.05)

        # DLPF_CFG=3: approximately 44 Hz accelerometer and 42 Hz gyro.
        self.bus.write_byte_data(address, self.CONFIG, 0x03)
        self.bus.write_byte_data(address, self.GYRO_CONFIG, 0x00)
        self.bus.write_byte_data(address, self.ACCEL_CONFIG, 0x00)

    @staticmethod
    def _signed_16(high: int, low: int) -> int:
        value = (high << 8) | low
        return value - 65536 if value & 0x8000 else value

    def read(self) -> ImuSample:
        data = self.bus.read_i2c_block_data(
            self.address,
            self.ACCEL_XOUT_H,
            14,
        )
        ax = self._signed_16(data[0], data[1])
        ay = self._signed_16(data[2], data[3])
        az = self._signed_16(data[4], data[5])
        gx = self._signed_16(data[8], data[9])
        gy = self._signed_16(data[10], data[11])
        gz = self._signed_16(data[12], data[13])

        del gx, gy  # Only planar gyro-Z is needed here.
        accel_scale = 9.80665 / 16384.0
        gyro_scale = math.pi / (180.0 * 131.0)

        return ImuSample(
            gyro_z_rad_s=gz * gyro_scale,
            accel_x_m_s2=ax * accel_scale,
            accel_y_m_s2=ay * accel_scale,
            accel_z_m_s2=az * accel_scale,
        )


def example_loop(read_all_encoder_counts) -> None:
    """Example integration; replace read_all_encoder_counts for your hardware."""
    config = OdometryConfig(
        wheel_radius_m=0.050,
        rear_track_width_m=0.290,
        ticks_per_revolution=2048,
        encoder_signs={
            "front_left": 1.0,
            "front_right": 1.0,
            "rear_left": 1.0,
            "rear_right": 1.0,
        },
    )
    imu_device = MPU6050(bus_number=1, address=0x68)
    estimator = F110WheelImuOdometry(config)

    # Vehicle must remain motionless during this calibration.
    calibration_samples = []
    for _ in range(500):
        calibration_samples.append(imu_device.read().gyro_z_rad_s)
        time.sleep(0.005)
    bias = estimator.calibrate_stationary_gyro(calibration_samples)
    print(f"Initial gyro bias: {bias:+.6f} rad/s")

    period_s = 0.01  # 100 Hz
    next_deadline = time.monotonic()
    while True:
        timestamp = time.monotonic()
        counts = read_all_encoder_counts()
        imu_sample = imu_device.read()
        state = estimator.update(timestamp, counts, imu_sample)

        if state is not None:
            print(
                f"x={state.x_m:+.3f} m  y={state.y_m:+.3f} m  "
                f"yaw={math.degrees(state.yaw_rad):+.1f} deg  "
                f"v={state.speed_m_s:+.2f} m/s  "
                f"w={state.yaw_rate_rad_s:+.2f} rad/s  "
                f"bias={state.gyro_bias_rad_s:+.4f}  "
                f"stationary={state.stationary}  slip={state.slip_detected}"
            )

        next_deadline += period_s
        remaining = next_deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        else:
            next_deadline = time.monotonic()
