import os
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VariationalEncoderRGB(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()

        self.model_file = os.path.join(
            "autoencoder_rgb",
            "model",
            "var_encoder_rgb.pth",
        )

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 256 * 5 * 10

        self.shared = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

        self.kl = torch.tensor(0.0, device=DEVICE)

    def forward(self, x, sample=False):
        x = x.to(DEVICE)

        h = self.features(x)
        h = torch.flatten(h, start_dim=1)
        h = self.shared(h)

        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), -10.0, 10.0)

        if sample:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        self.kl = -0.5 * torch.mean(
            torch.sum(
                1.0 + logvar - mu.pow(2) - logvar.exp(),
                dim=1,
            )
        )

        return torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)

    def save(self):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(
            torch.load(self.model_file, map_location=DEVICE)
        )
        self.to(DEVICE)
        self.eval()
