import torch
import numpy as np
from autoencoder_rgb.encoder_rgb import VariationalEncoderRGB


class EncodeStateRGB:
    def __init__(self, latent_dim):
        self.latent_dim = int(latent_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_encoder = VariationalEncoderRGB(self.latent_dim).to(self.device)
        self.conv_encoder.load()
        self.conv_encoder.eval()
        for parameter in self.conv_encoder.parameters():
            parameter.requires_grad = False
        print("RGB encoder loaded | latent={} | device={}".format(self.latent_dim, self.device))

    def process(self, observation):
        image_np = np.asarray(observation[0])
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError("Ảnh RGB phải có shape (H,W,3), nhận được {}".format(image_np.shape))

        image_obs = torch.from_numpy(image_np.copy()).to(
            device=self.device, dtype=torch.float32
        ) / 255.0
        image_obs = image_obs.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            latent = self.conv_encoder(image_obs, sample=False)

        navigation_obs = torch.as_tensor(
            observation[1], dtype=torch.float32, device=self.device
        ).view(-1)

        state = torch.cat((latent.view(-1), navigation_obs), dim=0)
        return torch.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
