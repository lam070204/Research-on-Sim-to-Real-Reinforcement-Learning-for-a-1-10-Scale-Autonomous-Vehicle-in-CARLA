# -*- coding: utf-8 -*-
import os
import sys
import time
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from autoencoder_rgb.encoder_rgb_v5 import VariationalEncoderRGB
from autoencoder_rgb.decoder_rgb_v5 import DecoderRGB


# ================================================================
# VAE RGB V3 - TRAIN / AUTO RESUME TRỰC TIẾP MODEL_V3
#
# Lần đầu:
#   - Nếu chưa có checkpoint V3 -> train mới.
#
# Các lần sau:
#   - Nếu có model_v5/vae_rgb_v5_last.pth -> tự load và train tiếp.
#   - Dataset mới phải được cộng chung vào dataset_v3 để model học lại
#     cả data cũ + data map mới, tránh quên data cũ.
#   - Trước khi resume, script tự backup các checkpoint V3 hiện tại.
#
# QUAN TRỌNG:
#   PPO cũ đã train với encoder V3 cũ (policy54/70/71...) cần đúng
#   latent V3 cũ. Khi train chồng encoder V3, latent sẽ thay đổi.
#   Backup tự động bên dưới giúp giữ lại encoder cũ để rollback/benchmark.
# ================================================================

NUM_EPOCHS = 20          # Mỗi lần chạy sẽ train THÊM 20 epoch
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LATENT_SPACE = 95
KL_BETA = 1e-4

AUTO_RESUME = True
BACKUP_BEFORE_RESUME = True

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "autoencoder_rgb",
    "dataset_v3",
)
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT,
    "autoencoder_rgb",
)

# GHI/RESUME TRỰC TIẾP MODEL V3
MODEL_DIR = os.path.join(
    OUTPUT_ROOT,
    "model_v5",
)
RECON_DIR = os.path.join(
    OUTPUT_ROOT,
    "reconstructed_v5",
)
RUN_DIR = os.path.join(
    PROJECT_ROOT,
    "runs",
    "vae_rgb_v5",
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vae_rgb_v5_best.pth",
)
LAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "vae_rgb_v5_last.pth",
)
BEST_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "var_encoder_rgb_v5_best.pth",
)
BEST_DECODER_PATH = os.path.join(
    MODEL_DIR,
    "decoder_rgb_v5_best.pth",
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
            os.path.join(RECON_DIR, "epoch_{:03d}.png".format(epoch)),
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
            "Không tìm thấy train dataset: {}".format(TRAIN_DIR)
        )

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(
            "Không tìm thấy test dataset: {}".format(TEST_DIR)
        )

    train_count = count_images(TRAIN_DIR)
    test_count = count_images(TEST_DIR)

    if train_count <= 0:
        raise RuntimeError("Train dataset rỗng: {}".format(TRAIN_DIR))

    if test_count <= 0:
        raise RuntimeError("Test dataset rỗng: {}".format(TEST_DIR))

    print("=" * 72)
    print("VAE RGB V3 DATASET")
    print("=" * 72)
    print("Train :", train_count, "|", TRAIN_DIR)
    print("Test  :", test_count, "|", TEST_DIR)
    print("Total :", train_count + test_count)
    print("=" * 72)


def backup_current_v3():
    """Backup checkpoint V3 hiện tại trước khi ghi đè/resume."""
    paths = [
        BEST_MODEL_PATH,
        LAST_MODEL_PATH,
        BEST_ENCODER_PATH,
        BEST_DECODER_PATH,
    ]
    existing = [p for p in paths if os.path.isfile(p)]

    if not existing:
        return None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(
        OUTPUT_ROOT,
        "model_v5_backup_before_resume_" + stamp,
    )
    os.makedirs(backup_dir, exist_ok=True)

    for src in existing:
        shutil.copy2(
            src,
            os.path.join(backup_dir, os.path.basename(src)),
        )

    print("Backup V3 cũ:", backup_dir)
    return backup_dir


def load_resume_checkpoint(model, optimizer):
    """
    Ưu tiên LAST để tiếp tục đúng epoch + optimizer.
    Nếu LAST không có nhưng BEST full VAE có thì resume từ BEST.
    """
    if not AUTO_RESUME:
        return False, 1, None

    if os.path.isfile(LAST_MODEL_PATH):
        resume_path = LAST_MODEL_PATH
    elif os.path.isfile(BEST_MODEL_PATH):
        resume_path = BEST_MODEL_PATH
    else:
        return False, 1, None

    if BACKUP_BEFORE_RESUME:
        backup_current_v3()

    print("=" * 72)
    print("AUTO RESUME VAE RGB V3")
    print("=" * 72)
    print("Loading:", resume_path)

    checkpoint = torch.load(
        resume_path,
        map_location=DEVICE,
    )

    latent = int(checkpoint.get("latent_space", -1))
    if latent != LATENT_SPACE:
        raise RuntimeError(
            "Checkpoint latent_space={} nhưng code LATENT_SPACE={}".format(
                latent,
                LATENT_SPACE,
            )
        )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint không có model_state_dict: {}".format(resume_path)
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Cho phép sau này chỉ đổi LEARNING_RATE ở đầu file.
        # State optimizer vẫn được kế thừa, nhưng LR dùng giá trị hiện tại.
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE

    loaded_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = loaded_epoch + 1

    print("Loaded epoch :", loaded_epoch)
    print("Start epoch  :", start_epoch)
    print("Learning rate:", LEARNING_RATE)
    print("=" * 72)

    return True, start_epoch, resume_path


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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    resumed, start_epoch, loaded_path = load_resume_checkpoint(
        model,
        optimizer,
    )

    # Khi thêm map/data mới, val_loss cũ không còn là mốc công bằng.
    # Đánh giá V3 cũ trên TEST SET HIỆN TẠI trước khi fine-tune.
    # Chỉ ghi đè BEST nếu training mới thực sự tốt hơn baseline này.
    if resumed:
        baseline_val, baseline_recon, baseline_kl = run_epoch(
            model,
            test_loader,
            optimizer=None,
        )
        best_val = float(baseline_val)

        print(
            "RESUME BASELINE CURRENT DATASET | "
            "val={:.6f} | recon={:.6f} | kl={:.6f}".format(
                baseline_val,
                baseline_recon,
                baseline_kl,
            )
        )
    else:
        start_epoch = 1
        best_val = float("inf")
        print("Không có checkpoint V3 -> train mới từ đầu.")

    writer = SummaryWriter(RUN_DIR)
    started = time.time()
    end_epoch = start_epoch + NUM_EPOCHS - 1

    print("Device :", DEVICE)
    print("Train  :", len(train_data))
    print("Test   :", len(test_data))
    print("Model  :", MODEL_DIR)
    print("Resume :", loaded_path if resumed else "NO")
    print(
        "Epochs : {} -> {} (train thêm {} epoch)".format(
            start_epoch,
            end_epoch,
            NUM_EPOCHS,
        )
    )

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, train_recon, train_kl = run_epoch(
            model,
            train_loader,
            optimizer,
        )
        val_loss, val_recon, val_kl = run_epoch(
            model,
            test_loader,
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
            "train_loss": train_loss,
            "dataset_train_count": len(train_data),
            "dataset_test_count": len(test_data),
        }

        # LAST luôn là trạng thái mới nhất để lần sau resume tiếp.
        torch.save(checkpoint, LAST_MODEL_PATH)

        # BEST chỉ update khi tốt hơn V3 cũ trên dataset hiện tại.
        if val_loss < best_val:
            best_val = val_loss

            torch.save(checkpoint, BEST_MODEL_PATH)
            torch.save(
                model.encoder.state_dict(),
                BEST_ENCODER_PATH,
            )
            torch.save(
                model.decoder.state_dict(),
                BEST_DECODER_PATH,
            )
            best_tag = " | NEW BEST"
        else:
            best_tag = ""

        print(
            "EPOCH {}/{} | "
            "train={:.6f} | "
            "val={:.6f} | "
            "best={:.6f} | "
            "time={:.1f} min{}".format(
                epoch,
                end_epoch,
                train_loss,
                val_loss,
                best_val,
                (time.time() - started) / 60.0,
                best_tag,
            )
        )

    writer.close()

    print("=" * 72)
    print("VAE RGB V3 TRAIN COMPLETE")
    print("=" * 72)
    print("Best VAE :", BEST_MODEL_PATH)
    print("Last VAE :", LAST_MODEL_PATH)
    print("Encoder  :", BEST_ENCODER_PATH)
    print("Decoder  :", BEST_DECODER_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()

