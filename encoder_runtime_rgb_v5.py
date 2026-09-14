# -*- coding: utf-8 -*-
"""
Robust frozen RGB encoder V5.

Fix device mismatch:
- Model device is the single source of truth.
- Input tensor is created explicitly on CPU first.
- Immediately before forward(), input is forced onto the model device.
- Works even if another module changes PyTorch's default CUDA tensor type.

Input:
    RGB uint8/float numpy image, shape (80,160,3)

Output:
    latent mu, shape (95,)
"""

import os

import numpy as np
import torch

from autoencoder_rgb.encoder_rgb_v5 import VariationalEncoderRGB


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LATENT_DIM_V5 = 95
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 160
IMAGE_CHANNELS = 3

ENCODER_V5_PATH = os.path.join(
    PROJECT_ROOT,
    "autoencoder_rgb",
    "model_v5",
    "var_encoder_rgb_v5_best.pth",
)


class EncodeRGBV5:
    def __init__(self, latent_dim=LATENT_DIM_V5, device=None):
        self.latent_dim = int(latent_dim)

        if self.latent_dim != LATENT_DIM_V5:
            raise ValueError(
                "V5 latent_dim phải là {}, nhận được {}.".format(
                    LATENT_DIM_V5,
                    self.latent_dim,
                )
            )

        if device is None:
            requested_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            requested_device = torch.device(device)

        if not os.path.isfile(ENCODER_V5_PATH):
            raise FileNotFoundError(
                "Không tìm thấy encoder V5: {}".format(
                    ENCODER_V5_PATH
                )
            )

        self.encoder = VariationalEncoderRGB(
            self.latent_dim
        )

        state_dict = torch.load(
            ENCODER_V5_PATH,
            map_location=requested_device,
        )

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "encoder_state_dict" in state_dict:
                state_dict = state_dict["encoder_state_dict"]

        self.encoder.load_state_dict(
            state_dict,
            strict=True,
        )

        self.encoder = self.encoder.to(
            requested_device
        )

        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        # IMPORTANT:
        # Do not trust a separate cached device variable.
        # Always read actual model-parameter device.
        self.device = next(
            self.encoder.parameters()
        ).device

        print(
            "RGB V5 encoder runtime loaded | latent={} | device={} | model={}".format(
                self.latent_dim,
                self.device,
                ENCODER_V5_PATH,
            )
        )

    def _image_to_tensor(self, image_rgb):
        image_np = np.asarray(image_rgb)

        if image_np.ndim != 3:
            raise ValueError(
                "RGB V5 phải có HxWxC, nhận được shape {}.".format(
                    image_np.shape
                )
            )

        expected_shape = (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS,
        )

        if tuple(image_np.shape) != expected_shape:
            raise ValueError(
                "RGB V5 phải có shape {}, nhận được {}.".format(
                    expected_shape,
                    tuple(image_np.shape),
                )
            )

        if np.issubdtype(
            image_np.dtype,
            np.floating,
        ):
            image_np = np.nan_to_num(
                image_np,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            if float(image_np.max()) <= 1.0:
                image_np = image_np * 255.0

        image_np = np.clip(
            image_np,
            0.0,
            255.0,
        ).astype(
            np.float32,
            copy=False,
        )

        # Force CPU construction explicitly.
        # This avoids surprises if some imported CARLA/PyTorch code
        # changed the global default tensor type to CUDA.
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_np)
        ).cpu()

        image_tensor = (
            image_tensor
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
            .float()
            .div(255.0)
        )

        # Model device is authoritative.
        model_device = next(
            self.encoder.parameters()
        ).device

        image_tensor = image_tensor.to(
            device=model_device,
            dtype=torch.float32,
            non_blocking=False,
        )

        return image_tensor

    @torch.no_grad()
    def encode(self, image_rgb):
        model_device = next(
            self.encoder.parameters()
        ).device

        image_tensor = self._image_to_tensor(
            image_rgb
        )

        # Defensive final check immediately before forward().
        if image_tensor.device != model_device:
            image_tensor = image_tensor.to(
                model_device
            )

        latent = self.encoder(
            image_tensor,
            sample=False,
        )

        latent = latent.reshape(-1)

        if latent.numel() != self.latent_dim:
            raise RuntimeError(
                "Encoder phải trả {} latent, nhận được {}.".format(
                    self.latent_dim,
                    latent.numel(),
                )
            )

        if not torch.isfinite(latent).all():
            raise RuntimeError(
                "Encoder tạo latent NaN/Inf."
            )

        return latent

    @property
    def frozen(self):
        return all(
            not parameter.requires_grad
            for parameter in self.encoder.parameters()
        )
