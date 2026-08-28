# -*- coding: utf-8 -*-
"""
GĐ10.3 - Observation builder V3

Observation layout (100 dims):
    [0:95]   VAE latent mu
    [95]     speed_mps
    [96]     yaw_rate_rad_s
    [97]     longitudinal_accel_mps2
    [98]     prev_steer_cmd
    [99]     prev_speed_cmd_mps

IMPORTANT:
- GĐ10.3 chỉ chốt cấu trúc và thứ tự observation.
- speed/yaw/accel ở đây vẫn là physical values chưa normalize.
- prev_steer_cmd được kỳ vọng đã clamp/rate-limit trong [-1, +1].
- prev_speed_cmd_mps là lệnh speed thực tế đã gửi, đơn vị m/s real-equivalent.
- Normalization yaw/accel sẽ được chốt sau bằng số đo real,
  không đoán hằng số ở bước này.
"""

import numpy as np
import torch

from encoder_runtime_rgb_v3 import EncodeRGBV3, LATENT_DIM_V3


OBSERVATION_DIM_V3 = 100
PROPRIO_DIM_V3 = 5

IDX_SPEED = 95
IDX_YAW_RATE = 96
IDX_LONG_ACCEL = 97
IDX_PREV_STEER = 98
IDX_PREV_SPEED = 99


class ObservationBuilderRGBV3:
    def __init__(self, encoder=None):
        self.encoder = encoder if encoder is not None else EncodeRGBV3()

        if self.encoder.latent_dim != LATENT_DIM_V3:
            raise ValueError(
                "Encoder latent phải là {}, nhận được {}.".format(
                    LATENT_DIM_V3,
                    self.encoder.latent_dim,
                )
            )

    def build(
        self,
        image_rgb,
        speed_mps,
        yaw_rate_rad_s,
        longitudinal_accel_mps2,
        prev_steer_cmd,
        prev_speed_cmd_mps,
    ):
        latent = self.encoder.encode(image_rgb)

        if tuple(latent.shape) != (LATENT_DIM_V3,):
            raise RuntimeError(
                "Latent shape sai: {}".format(tuple(latent.shape))
            )

        values = np.asarray(
            [
                speed_mps,
                yaw_rate_rad_s,
                longitudinal_accel_mps2,
                prev_steer_cmd,
                prev_speed_cmd_mps,
            ],
            dtype=np.float32,
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Proprioception chứa NaN/Inf: {}".format(values)
            )

        if not (-1.0 <= float(prev_steer_cmd) <= 1.0):
            raise ValueError(
                "prev_steer_cmd phải nằm trong [-1,1], nhận được {}.".format(
                    prev_steer_cmd
                )
            )

        proprio = torch.as_tensor(
            values,
            dtype=torch.float32,
            device=latent.device,
        )

        observation = torch.cat((latent, proprio), dim=0)

        if tuple(observation.shape) != (OBSERVATION_DIM_V3,):
            raise RuntimeError(
                "Observation V3 phải có shape ({},), nhận được {}.".format(
                    OBSERVATION_DIM_V3,
                    tuple(observation.shape),
                )
            )

        if not torch.isfinite(observation).all():
            raise RuntimeError("Observation V3 chứa NaN/Inf.")

        return observation

    @staticmethod
    def unpack_proprio(observation):
        if isinstance(observation, torch.Tensor):
            obs = observation.detach().cpu().numpy()
        else:
            obs = np.asarray(observation)

        if tuple(obs.shape) != (OBSERVATION_DIM_V3,):
            raise ValueError(
                "Observation phải có shape ({},), nhận được {}.".format(
                    OBSERVATION_DIM_V3,
                    tuple(obs.shape),
                )
            )

        return {
            "speed_mps": float(obs[IDX_SPEED]),
            "yaw_rate_rad_s": float(obs[IDX_YAW_RATE]),
            "longitudinal_accel_mps2": float(obs[IDX_LONG_ACCEL]),
            "prev_steer_cmd": float(obs[IDX_PREV_STEER]),
            "prev_speed_cmd_mps": float(obs[IDX_PREV_SPEED]),
        }
