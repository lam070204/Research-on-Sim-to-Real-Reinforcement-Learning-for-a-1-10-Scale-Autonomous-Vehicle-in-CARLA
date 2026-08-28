# -*- coding: utf-8 -*-
"""
GĐ4 - Action Controller V3 (calibration-ready baseline)

Mục tiêu
--------
PPO / calibration profile đưa vào:
    action = [steer_cmd, speed_cmd_mps]

Trong đó:
    steer_cmd      ∈ [-1, +1]
    speed_cmd_mps  ∈ [0, REAL.max_speed_mps]  (real-equivalent m/s)

Controller chuyển:
    speed_cmd -> throttle / brake của CARLA
    steer_cmd -> steer của CARLA

QUAN TRỌNG CHO GĐ4
------------------
- Không "nhét" đường đáp ứng REAL vào controller để ép SIM giống REAL.
- REAL log chỉ là target để SO SÁNH.
- Bắt đầu từ baseline này, chạy 4 test CARLA, rồi tune:
    1) longitudinal: physics + speed controller
    2) lateral/yaw: steering mapping + tire/physics
- Latency injection thuộc GĐ5, chưa làm ở đây.
- Domain randomization thuộc GĐ6, chưa làm ở đây.

Tương thích với carla_environment_rgb_v3.py hiện tại:
- ActionControllerV3(...)
- reset()
- step(steer_cmd, speed_cmd_mps, dt)
- prev_steer_cmd
- prev_speed_cmd_mps
"""

import math
import numpy as np

from simulation.carla_connection_v3 import carla
from vehicle_specs_v3 import (
    REAL,
    CARLA_GEOMETRY_SCALE,
    CARLA_FRONT_MAX_STEER_DEG,
)


# =============================================================================
# Action contract
# =============================================================================

CARLA_SCALE = float(CARLA_GEOMETRY_SCALE)
REAL_MAX_SPEED_MPS = float(REAL.max_speed_mps)

STEER_CMD_MIN = -1.0
STEER_CMD_MAX = +1.0

SPEED_CMD_MIN_MPS = 0.0
SPEED_CMD_MAX_MPS = REAL_MAX_SPEED_MPS


# =============================================================================
# GĐ4 calibration knobs
# =============================================================================
#
# Bắt đầu bằng None để command profile REAL và SIM giống nhau.
# Sau khi có phép đo servo/speed-command response rõ ràng mới freeze rate limit.
#
DEFAULT_STEER_RATE_LIMIT_PER_S = None
DEFAULT_SPEED_RATE_LIMIT_MPS2 = None

# Baseline feed-forward hiện có từ smoke-test CARLA.
# Đây CHƯA PHẢI giá trị GĐ4 final.
DEFAULT_FF_OFFSET = 0.22
DEFAULT_FF_SLOPE = 0.59

# Baseline PI feedback.
# Đây CHƯA PHẢI gain GĐ4 final.
DEFAULT_KP_THROTTLE = 0.85
DEFAULT_KI_THROTTLE = 0.22
DEFAULT_KP_BRAKE = 1.60

DEFAULT_INTEGRAL_LIMIT = 0.60

# Nếu CARLA nhanh hơn target quá deadband này thì bắt đầu brake.
DEFAULT_BRAKE_DEADBAND_REAL_MPS = 0.025

# Khi overspeed, không reset integral cứng mỗi tick.
DEFAULT_INTEGRAL_DECAY_OVERSPEED = 0.97

# Khi PPO/profile giảm target đủ lớn, bỏ integral của target cũ.
DEFAULT_TARGET_DROP_RESET_MPS = 0.08

# Stop behavior hiện vẫn là baseline.
# GĐ4 Test 3 sẽ quyết định giá trị final để khớp REAL:
#   0.5 -> 0 : ~0.72 s
#   0.7 -> 0 : ~1.24 s
DEFAULT_STOP_BRAKE = 1.0


# =============================================================================
# Helpers
# =============================================================================

def _clip(value, low, high):
    return float(np.clip(float(value), float(low), float(high)))


def vehicle_planar_speed_carla_mps(vehicle):
    """
    Raw CARLA planar speed [m/s].
    """
    velocity = vehicle.get_velocity()

    return float(
        math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
        )
    )


def vehicle_planar_speed_real_mps(
    vehicle,
    carla_scale=CARLA_SCALE,
):
    """
    Real-equivalent speed exposed to PPO/calibration [m/s].
    """
    scale = float(carla_scale)

    if scale <= 0.0:
        raise ValueError("carla_scale phải > 0.")

    return float(
        vehicle_planar_speed_carla_mps(vehicle)
        / scale
    )


# =============================================================================
# Action Controller
# =============================================================================

class ActionControllerV3:
    def __init__(
        self,
        vehicle,
        carla_scale=CARLA_SCALE,
        real_max_speed_mps=REAL_MAX_SPEED_MPS,
        steer_rate_limit_per_s=DEFAULT_STEER_RATE_LIMIT_PER_S,
        speed_rate_limit_mps2=DEFAULT_SPEED_RATE_LIMIT_MPS2,
        ff_offset=DEFAULT_FF_OFFSET,
        ff_slope=DEFAULT_FF_SLOPE,
        kp_throttle=DEFAULT_KP_THROTTLE,
        ki_throttle=DEFAULT_KI_THROTTLE,
        kp_brake=DEFAULT_KP_BRAKE,
        integral_limit=DEFAULT_INTEGRAL_LIMIT,
        brake_deadband_real_mps=DEFAULT_BRAKE_DEADBAND_REAL_MPS,
        integral_decay_overspeed=DEFAULT_INTEGRAL_DECAY_OVERSPEED,
        target_drop_reset_mps=DEFAULT_TARGET_DROP_RESET_MPS,
        stop_brake=DEFAULT_STOP_BRAKE,
    ):
        self.vehicle = vehicle

        self.carla_scale = float(carla_scale)
        self.real_max_speed_mps = float(real_max_speed_mps)

        if self.carla_scale <= 0.0:
            raise ValueError("carla_scale phải > 0.")

        if self.real_max_speed_mps <= 0.0:
            raise ValueError("real_max_speed_mps phải > 0.")

        self.steer_rate_limit_per_s = (
            None
            if steer_rate_limit_per_s is None
            else float(steer_rate_limit_per_s)
        )

        self.speed_rate_limit_mps2 = (
            None
            if speed_rate_limit_mps2 is None
            else float(speed_rate_limit_mps2)
        )

        if (
            self.steer_rate_limit_per_s is not None
            and self.steer_rate_limit_per_s <= 0.0
        ):
            raise ValueError(
                "steer_rate_limit_per_s phải > 0 hoặc None."
            )

        if (
            self.speed_rate_limit_mps2 is not None
            and self.speed_rate_limit_mps2 <= 0.0
        ):
            raise ValueError(
                "speed_rate_limit_mps2 phải > 0 hoặc None."
            )

        self.ff_offset = float(ff_offset)
        self.ff_slope = float(ff_slope)

        self.kp_throttle = float(kp_throttle)
        self.ki_throttle = float(ki_throttle)
        self.kp_brake = float(kp_brake)

        self.integral_limit = abs(float(integral_limit))

        self.brake_deadband_real_mps = abs(
            float(brake_deadband_real_mps)
        )

        self.integral_decay_overspeed = _clip(
            integral_decay_overspeed,
            0.0,
            1.0,
        )

        self.target_drop_reset_mps = abs(
            float(target_drop_reset_mps)
        )

        self.stop_brake = _clip(
            stop_brake,
            0.0,
            1.0,
        )

        # "prev_*" là command thực tế adapter đã áp ở tick trước.
        # Observation V3 đang dùng đúng hai giá trị này.
        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0

        self._speed_error_integral = 0.0
        self._last_target_speed_mps = 0.0

    # -------------------------------------------------------------------------
    # State/reset
    # -------------------------------------------------------------------------

    def reset(self):
        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0

        self._speed_error_integral = 0.0
        self._last_target_speed_mps = 0.0

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )

    # -------------------------------------------------------------------------
    # Command sanitization
    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_rate_limit(
        requested,
        previous,
        max_rate_per_s,
        dt,
    ):
        """
        Optional first-order slew/rate limiter.

        GĐ4 baseline:
            max_rate_per_s=None
        => giữ nguyên command profile để REAL và SIM nhận cùng command.

        Sau khi đo/freeze actuator response mới bật nếu cần.
        """
        requested = float(requested)
        previous = float(previous)
        dt = float(dt)

        if max_rate_per_s is None:
            return requested

        if dt <= 0.0:
            return previous

        max_delta = float(max_rate_per_s) * dt

        delta = _clip(
            requested - previous,
            -max_delta,
            +max_delta,
        )

        return float(previous + delta)

    def sanitize_commands(
        self,
        steer_cmd,
        speed_cmd_mps,
        dt,
    ):
        requested_steer = _clip(
            steer_cmd,
            STEER_CMD_MIN,
            STEER_CMD_MAX,
        )

        requested_speed = _clip(
            speed_cmd_mps,
            SPEED_CMD_MIN_MPS,
            self.real_max_speed_mps,
        )

        applied_steer = self._apply_rate_limit(
            requested=requested_steer,
            previous=self.prev_steer_cmd,
            max_rate_per_s=self.steer_rate_limit_per_s,
            dt=dt,
        )

        applied_speed_cmd = self._apply_rate_limit(
            requested=requested_speed,
            previous=self.prev_speed_cmd_mps,
            max_rate_per_s=self.speed_rate_limit_mps2,
            dt=dt,
        )

        applied_steer = _clip(
            applied_steer,
            STEER_CMD_MIN,
            STEER_CMD_MAX,
        )

        applied_speed_cmd = _clip(
            applied_speed_cmd,
            SPEED_CMD_MIN_MPS,
            self.real_max_speed_mps,
        )

        return applied_steer, applied_speed_cmd

    # -------------------------------------------------------------------------
    # Longitudinal controller
    # -------------------------------------------------------------------------

    def _throttle_feedforward(
        self,
        target_speed_real_mps,
    ):
        """
        Baseline CARLA feed-forward.

        Không dùng motor_pwm REAL làm CARLA throttle.
        GĐ4 sẽ tune lại hàm này/physics sau khi có SIM log.
        """
        target = float(target_speed_real_mps)

        if target <= 1e-4:
            return 0.0

        return _clip(
            self.ff_offset
            + self.ff_slope * target,
            0.0,
            1.0,
        )

    def _compute_speed_control(
        self,
        target_speed_real_mps,
        dt,
    ):
        current_speed_real = (
            vehicle_planar_speed_real_mps(
                self.vehicle,
                self.carla_scale,
            )
        )

        target_speed_real = _clip(
            target_speed_real_mps,
            SPEED_CMD_MIN_MPS,
            self.real_max_speed_mps,
        )

        error_real = (
            target_speed_real
            - current_speed_real
        )

        # ---------------------------------------------------------------------
        # Command = 0
        # ---------------------------------------------------------------------
        #
        # Đây là baseline cần được đo bằng GĐ4 Test 3.
        # Không coi brake=1.0 là "final real dynamics".
        #
        if target_speed_real <= 1e-4:
            self._speed_error_integral = 0.0
            self._last_target_speed_mps = 0.0

            return {
                "throttle": 0.0,
                "brake": self.stop_brake,
                "current_speed_real_mps": current_speed_real,
                "target_speed_real_mps": target_speed_real,
                "error_real_mps": error_real,
                "throttle_ff": 0.0,
            }

        # Target giảm rõ: bỏ windup của operating point cũ.
        if (
            target_speed_real
            < self._last_target_speed_mps
            - self.target_drop_reset_mps
        ):
            self._speed_error_integral = 0.0

        self._last_target_speed_mps = (
            target_speed_real
        )

        throttle_ff = (
            self._throttle_feedforward(
                target_speed_real
            )
        )

        # ---------------------------------------------------------------------
        # Overspeed
        # ---------------------------------------------------------------------
        if (
            error_real
            < -self.brake_deadband_real_mps
        ):
            self._speed_error_integral *= (
                self.integral_decay_overspeed
            )

            brake = _clip(
                self.kp_brake
                * (-error_real),
                0.0,
                1.0,
            )

            return {
                "throttle": 0.0,
                "brake": brake,
                "current_speed_real_mps": current_speed_real,
                "target_speed_real_mps": target_speed_real,
                "error_real_mps": error_real,
                "throttle_ff": throttle_ff,
            }

        # ---------------------------------------------------------------------
        # PI throttle
        # ---------------------------------------------------------------------
        dt = float(dt)

        if dt > 0.0:
            self._speed_error_integral += (
                error_real * dt
            )

            self._speed_error_integral = _clip(
                self._speed_error_integral,
                -self.integral_limit,
                +self.integral_limit,
            )

        throttle = (
            throttle_ff
            + self.kp_throttle * error_real
            + self.ki_throttle
            * self._speed_error_integral
        )

        throttle = _clip(
            throttle,
            0.0,
            1.0,
        )

        return {
            "throttle": throttle,
            "brake": 0.0,
            "current_speed_real_mps": current_speed_real,
            "target_speed_real_mps": target_speed_real,
            "error_real_mps": error_real,
            "throttle_ff": throttle_ff,
        }

    # -------------------------------------------------------------------------
    # Main step
    # -------------------------------------------------------------------------

    def step(
        self,
        steer_cmd,
        speed_cmd_mps,
        dt,
    ):
        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                "dt phải finite và > 0, nhận được {}.".format(dt)
            )

        requested_steer = float(steer_cmd)
        requested_speed = float(speed_cmd_mps)

        if not np.isfinite(requested_steer):
            raise ValueError("steer_cmd chứa NaN/Inf.")

        if not np.isfinite(requested_speed):
            raise ValueError("speed_cmd_mps chứa NaN/Inf.")

        (
            applied_steer,
            applied_speed_cmd,
        ) = self.sanitize_commands(
            steer_cmd=requested_steer,
            speed_cmd_mps=requested_speed,
            dt=dt,
        )

        speed_control = (
            self._compute_speed_control(
                target_speed_real_mps=(
                    applied_speed_cmd
                ),
                dt=dt,
            )
        )

        throttle = float(
            speed_control["throttle"]
        )

        brake = float(
            speed_control["brake"]
        )

        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                steer=float(applied_steer),
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
        )

        # Observation V3 dùng applied command ở tick trước.
        self.prev_steer_cmd = float(
            applied_steer
        )

        self.prev_speed_cmd_mps = float(
            applied_speed_cmd
        )

        current_speed_real = float(
            speed_control[
                "current_speed_real_mps"
            ]
        )

        target_speed_real = float(
            speed_control[
                "target_speed_real_mps"
            ]
        )

        speed_error_real = float(
            speed_control[
                "error_real_mps"
            ]
        )

        throttle_ff = float(
            speed_control[
                "throttle_ff"
            ]
        )

        # Telemetry cố ý phong phú để GĐ4 logger có thể ghi trực tiếp.
        return {
            # Command requested by PPO / GĐ4 profile
            "requested_steer_cmd": requested_steer,
            "requested_speed_cmd_mps": requested_speed,

            # Command actually applied after optional rate limit
            "actual_steer_cmd": float(
                applied_steer
            ),
            "actual_speed_cmd_mps": float(
                applied_speed_cmd
            ),

            # Alias rõ nghĩa hơn cho calibration code mới
            "applied_steer_cmd": float(
                applied_steer
            ),
            "applied_speed_cmd_mps": float(
                applied_speed_cmd
            ),

            # Steering physical-equivalent reference
            "applied_steer_angle_deg": float(
                applied_steer
                * CARLA_FRONT_MAX_STEER_DEG
            ),

            # Speed
            "current_speed_real_equiv_mps": (
                current_speed_real
            ),
            "current_speed_carla_mps": float(
                current_speed_real
                * self.carla_scale
            ),
            "target_speed_real_mps": (
                target_speed_real
            ),
            "target_speed_carla_mps": float(
                target_speed_real
                * self.carla_scale
            ),
            "speed_error_real_mps": (
                speed_error_real
            ),

            # Controller internal state
            "speed_error_integral": float(
                self._speed_error_integral
            ),
            "throttle_ff": throttle_ff,

            # CARLA low-level control
            "throttle": throttle,
            "brake": brake,

            # Metadata useful for GĐ4 logs
            "carla_scale": float(
                self.carla_scale
            ),
            "dt_s": dt,
        }
