import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from encoder_rgb import VariationalEncoderRGB
from decoder_rgb import DecoderRGB


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.path.join(
    "autoencoder",
    "dataset_rgb_autopilot",
    "test",
)

MODEL_PATH = os.path.join(
    "autoencoder_rgb",
    "model",
    "vae_rgb_best.pth",
)

OUTPUT_DIR = os.path.join(
    "autoencoder_rgb",
    "reconstructed",
    "final_test",
)


class VariationalAutoencoderRGB(torch.nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoderRGB(latent_dims)
        self.decoder = DecoderRGB(latent_dims)

    def forward(self, x):
        return self.decoder(self.encoder(x, sample=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    data = datasets.ImageFolder(DATA_DIR, transform=tf)
    loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=0)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    latent = checkpoint.get("latent_space", 95)

    model = VariationalAutoencoderRGB(latent).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= 10:
                break

            images = images.to(DEVICE)
            x_hat = model(images)

            comparison = torch.cat(
                [images.cpu(), x_hat.cpu()],
                dim=0,
            )

            path = os.path.join(
                OUTPUT_DIR,
                f"comparison_{i:03d}.png",
            )

            save_image(comparison, path, nrow=16)
            print("Saved:", path)


if __name__ == "__main__":
    main()
