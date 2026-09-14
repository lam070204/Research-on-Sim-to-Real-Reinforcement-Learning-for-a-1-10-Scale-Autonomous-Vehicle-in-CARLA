# -*- coding: utf-8 -*-
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''python .\train_ppo_rgb_v3.py --train true --load-checkpoint true --checkpoint-every-steps 20000'''
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autoencoder_rgb.encoder_rgb_v3 import VariationalEncoderRGB
from autoencoder_rgb.decoder_rgb_v3 import DecoderRGB

''' 
python autoencoder_rgb\train_vae_rgb_v3.py

'''
NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LATENT_SPACE = 95
KL_BETA = 1e-4

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "autoencoder_rgb",
    "dataset_v4",
)
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "autoencoder_rgb",
)
MODEL_DIR = os.path.join(
    OUTPUT_ROOT,
    "model_v3",
)
RECON_DIR = os.path.join(
    OUTPUT_ROOT,
    "reconstructed_v3",
)
RUN_DIR = os.path.join(
    PROJECT_ROOT,
    "runs",
    "vae_rgb_v3",
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vae_rgb_v3_best.pth",
)
LAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vae_rgb_v3_last.pth",
)
BEST_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "var_encoder_rgb_v3_best.pth",
)
BEST_DECODER_PATH = os.path.join(
    MODEL_DIR,
    "decoder_rgb_v3_best.pth",
)

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



def count_images(folder):
    total = 0

    if not os.path.isdir(folder):
        return 0

    for current_root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp")
            ):
                total += 1

    return total


def validate_dataset():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            "Không tìm thấy train dataset: {}".format(
                TRAIN_DIR
            )
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            "Không tìm thấy test dataset: {}".format(
                TEST_DIR
            )
        )

    train_count = count_images(TRAIN_DIR)
    test_count = count_images(TEST_DIR)

    if train_count <= 0:
        raise RuntimeError(
            "Train dataset rỗng: {}".format(TRAIN_DIR)
        )

    if test_count <= 0:
        raise RuntimeError(
            "Test dataset rỗng: {}".format(TEST_DIR)
        )

    print("=" * 72)
    print("VAE RGB V3 DATASET")
    print("=" * 72)
    print("Train :", train_count, "|", TRAIN_DIR)
    print("Test  :", test_count, "|", TEST_DIR)
    print("Total :", train_count + test_count)
    print("=" * 72)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    validate_dataset()

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

            # KHÔNG gọi model.encoder.save()/decoder.save():
            # các method đó đang trỏ về autoencoder_rgb/model (model cũ).
            # V3 phải lưu riêng để không ghi đè V2.
            torch.save(
                model.encoder.state_dict(),
                BEST_ENCODER_PATH,
            )
            torch.save(
                model.decoder.state_dict(),
                BEST_DECODER_PATH,
            )

        print(
            f"EPOCH {epoch}/{NUM_EPOCHS} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"best={best_val:.6f} | "
            f"time={(time.time()-started)/60:.1f} min"
        )

    writer.close()

    print("Best VAE :", BEST_MODEL_PATH)
    print("Last VAE :", LAST_MODEL_PATH)
    print("Encoder  :", BEST_ENCODER_PATH)
    print("Decoder  :", BEST_DECODER_PATH)


if __name__ == "__main__":
    main()
