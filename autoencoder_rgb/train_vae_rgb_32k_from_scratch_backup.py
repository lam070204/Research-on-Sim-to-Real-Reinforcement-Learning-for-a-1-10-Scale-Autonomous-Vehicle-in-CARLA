# -*- coding: utf-8 -*-
"""
Train VAE RGB từ đầu bằng cả dữ liệu CŨ + MỚI (32.000 ảnh).

Dữ liệu:
- CŨ: autoencoder-semantic/dataset_rgb_autopilot
- MỚI: RGB_DATA_COLLECTION/dataset_new_16000

An toàn:
- Không load và không ghi đè VAE/encoder cũ.
- Không gọi encoder.save() hoặc decoder.save().
- Lưu toàn bộ model mới vào:
  autoencoder_rgb/model_32k_from_scratch_backup
"""

import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from encoder_rgb import VariationalEncoderRGB
from decoder_rgb import DecoderRGB


NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 95
KL_BETA = 1e-4

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OLD_DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "autoencoder-semantic",
    "dataset_rgb_autopilot",
)
NEW_DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "RGB_DATA_COLLECTION",
    "dataset_new_16000",
)

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "autoencoder_rgb",
)
MODEL_DIR = os.path.join(
    OUTPUT_ROOT,
    "model_32k_from_scratch_backup",
)
RECON_DIR = os.path.join(
    OUTPUT_ROOT,
    "reconstructed_32k_from_scratch_backup",
)
RUN_DIR = os.path.join(
    PROJECT_ROOT,
    "runs",
    "vae_rgb_32k_from_scratch_backup",
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vae_rgb_32k_from_scratch_best.pth",
)
LAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vae_rgb_32k_from_scratch_last.pth",
)
ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "var_encoder_rgb_32k_from_scratch.pth",
)
DECODER_PATH = os.path.join(
    MODEL_DIR,
    "decoder_rgb_32k_from_scratch.pth",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class VariationalAutoencoderRGB(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoderRGB(latent_dims)
        self.decoder = DecoderRGB(latent_dims)

    def forward(self, x, sample=True):
        z = self.encoder(x, sample=sample)
        return self.decoder(z)


def compute_loss(x_hat, x, kl):
    recon = nn.functional.mse_loss(
        x_hat,
        x,
        reduction="mean",
    )
    total = recon + KL_BETA * kl
    return total, recon, kl


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total = 0.0
    recon_total = 0.0
    kl_total = 0.0
    count = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for images, _ in loader:
            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad()

            x_hat = model(
                images,
                sample=training,
            )

            loss, recon, kl = compute_loss(
                x_hat,
                images,
                model.encoder.kl,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

            batch = images.size(0)
            count += batch
            total += loss.item() * batch
            recon_total += recon.item() * batch
            kl_total += kl.item() * batch

    count = max(count, 1)

    return (
        total / count,
        recon_total / count,
        kl_total / count,
    )


def save_preview(model, loader, epoch):
    model.eval()

    with torch.no_grad():
        images, _ = next(iter(loader))
        images = images[:8].to(DEVICE)

        x_hat = model(
            images,
            sample=False,
        )

        comparison = torch.cat(
            [images.cpu(), x_hat.cpu()],
            dim=0,
        )

        save_image(
            comparison,
            os.path.join(
                RECON_DIR,
                "epoch_{:03d}.png".format(epoch),
            ),
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


def build_image_folder(root, split, transform):
    split_dir = os.path.join(root, split)

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            "Không tìm thấy: {}".format(split_dir)
        )

    return datasets.ImageFolder(
        split_dir,
        transform=transform,
    )


def main():
    expected = {
        ("OLD", "train"): 14400,
        ("OLD", "test"): 1600,
        ("NEW", "train"): 14400,
        ("NEW", "test"): 1600,
    }

    roots = {
        "OLD": OLD_DATA_ROOT,
        "NEW": NEW_DATA_ROOT,
    }

    print("\n===== KIỂM TRA DATASET 32K =====")

    for label, root in roots.items():
        for split in ("train", "test"):
            split_dir = os.path.join(root, split)
            actual = count_images(split_dir)
            target = expected[(label, split)]

            print(
                "{} {:5}: {} / {}".format(
                    label,
                    split.upper(),
                    actual,
                    target,
                )
            )

            if actual != target:
                raise RuntimeError(
                    "{} {} phải có {} ảnh, hiện có {}."
                    .format(
                        label,
                        split,
                        target,
                        actual,
                    )
                )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

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

    old_train = build_image_folder(
        OLD_DATA_ROOT,
        "train",
        train_tf,
    )
    new_train = build_image_folder(
        NEW_DATA_ROOT,
        "train",
        train_tf,
    )
    old_test = build_image_folder(
        OLD_DATA_ROOT,
        "test",
        test_tf,
    )
    new_test = build_image_folder(
        NEW_DATA_ROOT,
        "test",
        test_tf,
    )

    train_data = ConcatDataset(
        [old_train, new_train]
    )
    test_data = ConcatDataset(
        [old_test, new_test]
    )

    if len(train_data) != 28800:
        raise RuntimeError(
            "Train gộp phải có 28800 ảnh, hiện có {}."
            .format(len(train_data))
        )

    if len(test_data) != 3200:
        raise RuntimeError(
            "Test gộp phải có 3200 ảnh, hiện có {}."
            .format(len(test_data))
        )

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

    model = VariationalAutoencoderRGB(
        LATENT_SPACE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    writer = SummaryWriter(RUN_DIR)

    best_val = float("inf")
    started = time.time()

    print("\n===== TRAIN VAE 32K TỪ ĐẦU =====")
    print("Device       :", DEVICE)
    print("Old train    :", len(old_train))
    print("New train    :", len(new_train))
    print("Total train  :", len(train_data))
    print("Old test     :", len(old_test))
    print("New test     :", len(new_test))
    print("Total test   :", len(test_data))
    print("Epochs       :", NUM_EPOCHS)
    print("Batch size   :", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Latent       :", LATENT_SPACE)
    print("Output       :", MODEL_DIR)
    print("Không ghi đè model cũ.")
    print("=================================\n")

    try:
        for epoch in range(
            1,
            NUM_EPOCHS + 1,
        ):
            (
                train_loss,
                train_recon,
                train_kl,
            ) = run_epoch(
                model,
                train_loader,
                optimizer,
            )

            (
                val_loss,
                val_recon,
                val_kl,
            ) = run_epoch(
                model,
                test_loader,
            )

            writer.add_scalar(
                "Loss/train_total",
                train_loss,
                epoch,
            )
            writer.add_scalar(
                "Loss/val_total",
                val_loss,
                epoch,
            )
            writer.add_scalar(
                "Loss/train_recon",
                train_recon,
                epoch,
            )
            writer.add_scalar(
                "Loss/val_recon",
                val_recon,
                epoch,
            )
            writer.add_scalar(
                "Loss/train_kl",
                train_kl,
                epoch,
            )
            writer.add_scalar(
                "Loss/val_kl",
                val_kl,
                epoch,
            )

            save_preview(
                model,
                test_loader,
                epoch,
            )

            checkpoint = {
                "epoch": epoch,
                "latent_space": LATENT_SPACE,
                "model_state_dict": (
                    model.state_dict()
                ),
                "optimizer_state_dict": (
                    optimizer.state_dict()
                ),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "old_dataset": OLD_DATA_ROOT,
                "new_dataset": NEW_DATA_ROOT,
            }

            torch.save(
                checkpoint,
                LAST_MODEL_PATH,
            )

            if val_loss < best_val:
                best_val = val_loss

                torch.save(
                    checkpoint,
                    BEST_MODEL_PATH,
                )

                torch.save(
                    model.encoder.state_dict(),
                    ENCODER_PATH,
                )

                torch.save(
                    model.decoder.state_dict(),
                    DECODER_PATH,
                )

            print(
                "EPOCH {}/{} | "
                "train={:.6f} | "
                "val={:.6f} | "
                "best={:.6f} | "
                "time={:.1f} min"
                .format(
                    epoch,
                    NUM_EPOCHS,
                    train_loss,
                    val_loss,
                    best_val,
                    (
                        time.time() - started
                    ) / 60.0,
                )
            )

    finally:
        writer.close()

    print("\n===== TRAIN HOÀN TẤT =====")
    print("Best model :", BEST_MODEL_PATH)
    print("Last model :", LAST_MODEL_PATH)
    print("Encoder    :", ENCODER_PATH)
    print("Decoder    :", DECODER_PATH)
    print("Preview    :", RECON_DIR)
    print("")
    print(
        "Model cũ trong autoencoder_rgb/model "
        "vẫn còn nguyên."
    )
    print(
        "Model 32K mới chưa được kích hoạt cho PPO cũ."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng bởi người dùng.")
        print(
            "Checkpoint epoch hoàn tất gần nhất "
            "vẫn được giữ."
        )
