import os
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderRGB(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()

        self.model_file = os.path.join(
            "autoencoder_rgb",
            "model",
            "decoder_rgb.pth",
        )

        self.linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256 * 5 * 10),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.unflatten = nn.Unflatten(1, (256, 5, 10))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = z.to(DEVICE)
        x = self.linear(z)
        x = self.unflatten(x)
        return self.decoder(x)

    def save(self):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(
            torch.load(self.model_file, map_location=DEVICE)
        )
        self.to(DEVICE)
        self.eval()
