import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from encoder_rgb import VariationalEncoderRGB
from decoder_rgb import DecoderRGB


NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LATENT_SPACE = 95
KL_BETA = 1e-4

DATA_ROOT = os.path.join("autoencoder", "dataset_rgb_autopilot")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

OUTPUT_ROOT = "autoencoder_rgb"
MODEL_DIR = os.path.join(OUTPUT_ROOT, "model")
RECON_DIR = os.path.join(OUTPUT_ROOT, "reconstructed")
RUN_DIR = os.path.join("runs", "vae_rgb_autopilot")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "vae_rgb_best.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "vae_rgb_last.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VariationalAutoencoderRGB(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoderRGB(latent_dims)
        self.decoder = DecoderRGB(latent_dims)

    def forward(self, x, sample=True):
        z = self.encoder(x, sample=sample)
        return self.decoder(z)


def compute_loss(x_hat, x, kl):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    total = recon + KL_BETA * kl
    return total, recon, kl


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total = recon_total = kl_total = 0.0
    count = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, _ in loader:
            images = images.to(DEVICE)

            if training:
                optimizer.zero_grad()

            x_hat = model(images, sample=training)
            loss, recon, kl = compute_loss(
                x_hat, images, model.encoder.kl
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            batch = images.size(0)
            count += batch
            total += loss.item() * batch
            recon_total += recon.item() * batch
            kl_total += kl.item() * batch

    count = max(count, 1)
    return total / count, recon_total / count, kl_total / count


def save_preview(model, loader, epoch):
    model.eval()

    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images[:8].to(DEVICE)
        x_hat = model(images, sample=False)

        comparison = torch.cat(
            [images.cpu(), x_hat.cpu()],
            dim=0,
        )

        save_image(
            comparison,
            os.path.join(RECON_DIR, f"epoch_{epoch:03d}.png"),
            nrow=8,
        )


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.05,
        ),
        transforms.ToTensor(),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    test_data = datasets.ImageFolder(TEST_DIR, transform=test_tf)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = VariationalAutoencoderRGB(LATENT_SPACE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(RUN_DIR)

    best_val = float("inf")
    started = time.time()

    print("Device:", DEVICE)
    print("Train:", len(train_data))
    print("Test :", len(test_data))
    print("Output:", OUTPUT_ROOT)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_recon, train_kl = run_epoch(
            model, train_loader, optimizer
        )
        val_loss, val_recon, val_kl = run_epoch(
            model, test_loader
        )

        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/val_total", val_loss, epoch)
        writer.add_scalar("Loss/train_recon", train_recon, epoch)
        writer.add_scalar("Loss/val_recon", val_recon, epoch)
        writer.add_scalar("Loss/train_kl", train_kl, epoch)
        writer.add_scalar("Loss/val_kl", val_kl, epoch)

        save_preview(model, test_loader, epoch)

        checkpoint = {
            "epoch": epoch,
            "latent_space": LATENT_SPACE,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }

        torch.save(checkpoint, LAST_MODEL_PATH)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, BEST_MODEL_PATH)
            model.encoder.save()
            model.decoder.save()

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"best={best_val:.6f} | "
            f"time={(time.time()-started)/60:.1f} min"
        )

    writer.close()

    print("Best:", BEST_MODEL_PATH)
    print("Last:", LAST_MODEL_PATH)
    print("Encoder:", model.encoder.model_file)
    print("Decoder:", model.decoder.model_file)


if __name__ == "__main__":
    main()
