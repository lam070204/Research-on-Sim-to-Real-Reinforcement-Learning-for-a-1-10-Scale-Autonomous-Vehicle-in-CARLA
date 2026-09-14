# -*- coding: utf-8 -*-
"""
Shared command-space adapter for STABLE035.

Use the SAME class in CARLA training and Jetson deployment.

PPO contract:
    steer_unit in [-1, 1]
    speed_unit in [0, 1]

Output:
    steer_cmd in [-1, 1]
    speed_cmd_mps in [0, speed_max_mps]

Stability controls happen in logical command space BEFORE any simulated
actuator/dynamics randomization.
"""
from __future__ import print_function


def _clip(value, low, high):
    return float(max(float(low), min(float(high), float(value))))


class StableActionAdapterV5(object):
    def __init__(
        self,
        speed_max_mps=0.35,
        steer_deadband=0.015,
        steer_rate_limit_per_s=3.0,
        speed_rate_limit_mps2=0.50,
    ):
        self.speed_max_mps = max(0.0, float(speed_max_mps))
        self.steer_deadband = _clip(steer_deadband, 0.0, 0.25)
        self.steer_rate_limit_per_s = (
            None if steer_rate_limit_per_s is None
            else max(0.0, float(steer_rate_limit_per_s))
        )
        self.speed_rate_limit_mps2 = (
            None if speed_rate_limit_mps2 is None
            else max(0.0, float(speed_rate_limit_mps2))
        )
        self.reset()

    def reset(self):
        self.prev_steer_cmd = 0.0
        self.prev_speed_cmd_mps = 0.0

    def transform(self, steer_unit, speed_unit, dt):
        dt = max(0.0, float(dt))

        raw_steer = _clip(steer_unit, -1.0, 1.0)
        raw_speed_unit = _clip(speed_unit, 0.0, 1.0)

        steer_cmd = (
            0.0
            if abs(raw_steer) < float(self.steer_deadband)
            else raw_steer
        )
        speed_cmd = (
            raw_speed_unit
            * float(self.speed_max_mps)
        )

        if (
            self.steer_rate_limit_per_s is not None
            and dt > 0.0
        ):
            max_delta = (
                float(self.steer_rate_limit_per_s)
                * dt
            )
            steer_cmd = _clip(
                steer_cmd,
                self.prev_steer_cmd - max_delta,
                self.prev_steer_cmd + max_delta,
            )

        if (
            self.speed_rate_limit_mps2 is not None
            and dt > 0.0
        ):
            max_delta = (
                float(self.speed_rate_limit_mps2)
                * dt
            )
            speed_cmd = _clip(
                speed_cmd,
                self.prev_speed_cmd_mps - max_delta,
                self.prev_speed_cmd_mps + max_delta,
            )

        steer_cmd = _clip(
            steer_cmd,
            -1.0,
            1.0,
        )
        speed_cmd = _clip(
            speed_cmd,
            0.0,
            self.speed_max_mps,
        )

        self.prev_steer_cmd = float(steer_cmd)
        self.prev_speed_cmd_mps = float(speed_cmd)

        return {
            "raw_steer_unit": float(raw_steer),
            "raw_speed_unit": float(raw_speed_unit),
            "steer_cmd": float(steer_cmd),
            "speed_cmd_mps": float(speed_cmd),
        }
