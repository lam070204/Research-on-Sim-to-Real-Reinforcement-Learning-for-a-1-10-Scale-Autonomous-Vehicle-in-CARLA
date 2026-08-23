import sys
import torch
from autoencoder.encoder_rgb import VariationalEncoder


class EncodeState:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            self.conv_encoder = VariationalEncoder(
                self.latent_dim
            ).to(self.device)

            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False

        except Exception as e:
            print("Encoder could not be initialized.")
            print("REAL ERROR:", repr(e))
            raise

    def process(self, observation):

        image_obs = torch.tensor(
            observation[0],
            dtype=torch.float32
        ).to(self.device)

        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0, 3, 2, 1)

        # VAE Encoder
        with torch.no_grad():
            image_obs = self.conv_encoder(image_obs)

        # Navigation observation
        navigation_obs = torch.tensor(
            observation[1],
            dtype=torch.float32
        ).to(self.device)

        state = torch.cat(
            (
                image_obs.view(-1),
                navigation_obs.view(-1)
            ),
            dim=-1
        )

        state = torch.nan_to_num(
            state,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0
        )

        return state