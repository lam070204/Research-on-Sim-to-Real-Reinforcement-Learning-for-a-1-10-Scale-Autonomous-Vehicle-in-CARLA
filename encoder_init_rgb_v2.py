# -*- coding: utf-8 -*-
"""
Encoder RGB V2 dùng model VAE 32K train từ đầu.

Không thay đổi encoder_init_rgb.py cũ.
Không ghi đè model cũ.
"""

import os
import torch
import numpy as np

from autoencoder_rgb.encoder_rgb import VariationalEncoderRGB


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

ENCODER_V2_PATH = os.path.join(
    PROJECT_ROOT,
    "autoencoder_rgb",
    "model_32k_from_scratch_backup",
    "var_encoder_rgb_32k_from_scratch.pth",
)


class EncodeStateRGBV2:
    def __init__(self, latent_dim):
        self.latent_dim = int(latent_dim)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not os.path.isfile(ENCODER_V2_PATH):
            raise FileNotFoundError(
                "Không tìm thấy encoder V2: {}".format(
                    ENCODER_V2_PATH
                )
            )

        self.conv_encoder = VariationalEncoderRGB(
            self.latent_dim
        ).to(self.device)

        state_dict = torch.load(
            ENCODER_V2_PATH,
            map_location=self.device,
        )

        # Hỗ trợ một số định dạng checkpoint phổ biến.
        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
        ):
            state_dict = state_dict["state_dict"]

        self.conv_encoder.load_state_dict(
            state_dict,
            strict=True,
        )

        self.conv_encoder.eval()

        for parameter in self.conv_encoder.parameters():
            parameter.requires_grad = False

        print(
            "RGB V2 encoder loaded | latent={} | device={} | model={}"
            .format(
                self.latent_dim,
                self.device,
                ENCODER_V2_PATH,
            )
        )

    def process(self, observation):
        if not isinstance(observation, (list, tuple)):
            raise TypeError(
                "Observation phải là list/tuple gồm [ảnh RGB, navigation]."
            )

        if len(observation) < 2:
            raise ValueError(
                "Observation phải có ít nhất 2 phần tử."
            )

        image_np = np.asarray(observation[0])

        if (
            image_np.ndim != 3
            or image_np.shape[2] != 3
        ):
            raise ValueError(
                "Ảnh RGB phải có shape (H,W,3), nhận được {}"
                .format(image_np.shape)
            )

        if image_np.shape[:2] != (80, 160):
            raise ValueError(
                "Ảnh RGB V2 phải có kích thước 160x80, "
                "nhận được {}x{}."
                .format(
                    image_np.shape[1],
                    image_np.shape[0],
                )
            )

        image_obs = torch.from_numpy(
            image_np.copy()
        ).to(
            device=self.device,
            dtype=torch.float32,
        ) / 255.0

        image_obs = (
            image_obs
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        with torch.no_grad():
            latent = self.conv_encoder(
                image_obs,
                sample=False,
            )

        navigation_obs = torch.as_tensor(
            observation[1],
            dtype=torch.float32,
            device=self.device,
        ).view(-1)

        if navigation_obs.numel() != 5:
            raise ValueError(
                "Navigation V2 phải có đúng 5 giá trị, "
                "nhận được {}."
                .format(navigation_obs.numel())
            )

        state = torch.cat(
            (
                latent.view(-1),
                navigation_obs,
            ),
            dim=0,
        )

        if state.numel() != self.latent_dim + 5:
            raise RuntimeError(
                "State V2 phải có {} giá trị, nhận được {}."
                .format(
                    self.latent_dim + 5,
                    state.numel(),
                )
            )

        return torch.nan_to_num(
            state,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )
