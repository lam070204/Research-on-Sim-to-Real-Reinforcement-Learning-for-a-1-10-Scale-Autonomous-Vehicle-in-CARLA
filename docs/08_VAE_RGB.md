# 08 VAE RGB

## Variational Autoencoder cho RGB Camera

Tài liệu về VAE (Variational Autoencoder) được sử dụng để nén ảnh RGB từ camera thành latent vector.

## Tổng quan

VAE được sử dụng để giảm chiều dữ liệu ảnh RGB từ 38,400 pixels (160x80x3) xuống còn 95 chiều latent vector, giúp giảm đáng kể computational cost cho PPO training.

## Kiến trúc VAE

### Encoder

```
Input: 3x80x160 (RGB image)
↓
Conv2d(3, 32, 3, stride=2) + ReLU → 32x39x79
↓
Conv2d(32, 64, 3, stride=2) + ReLU → 64x18x38
↓
Conv2d(64, 128, 3, stride=2) + ReLU → 128x7x17
↓
Conv2d(128, 256, 3, stride=2) + ReLU → 256x2x7
↓
Flatten → 3584 dimensions
↓
Linear(3584, 95) → μ (mean)
Linear(3584, 95) → σ (std)
↓
z = μ + σ * ε  (reparameterization trick)
↓
Output: 95-dim latent vector
```

### Decoder

```
Input: 95-dim latent vector
↓
Linear(95, 3584) + ReLU
↓
Reshape → 256x2x7
↓
ConvTranspose2d(256, 128, 3, stride=2) + ReLU → 128x7x17
↓
ConvTranspose2d(128, 64, 3, stride=2) + ReLU → 64x18x38
↓
ConvTranspose2d(64, 32, 3, stride=2) + ReLU → 32x39x79
↓
ConvTranspose2d(32, 3, 3, stride=2) + Sigmoid → 3x80x160
↓
Output: Reconstructed RGB image
```

## Implementation

### Encoder Network

File: `autoencoder_rgb/encoder_rgb.py`

```python
import torch
import torch.nn as nn

class VariationalEncoderRGB(nn.Module):
    def __init__(self, latent_dim=95):
        super(VariationalEncoderRGB, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),  # 3x80x160 → 32x39x79
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),  # 32x39x79 → 64x18x38
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),  # 64x18x38 → 128x7x17
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2),  # 128x7x17 → 256x2x7
            nn.ReLU()
        )
        
        # Flatten: 256x2x7 = 3584
        self.flatten = nn.Flatten()
        
        # Latent space
        self.fc_mu = nn.Linear(3584, latent_dim)
        self.fc_logvar = nn.Linear(3584, latent_dim)
        
        self.latent_dim = latent_dim
    
    def forward(self, x):
        # x: (batch, 3, 80, 160)
        h = self.conv_layers(x)  # (batch, 256, 2, 7)
        h = self.flatten(h)  # (batch, 3584)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        
        return z, mu, logvar
```

### Decoder Network

File: `autoencoder_rgb/decoder_rgb.py`

```python
import torch
import torch.nn as nn

class DecoderRGB(nn.Module):
    def __init__(self, latent_dim=95):
        super(DecoderRGB, self).__init__()
        
        # From latent to flattened conv features
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 3584),
            nn.ReLU()
        )
        
        # Transposed convolutions
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),  # 256x2x7 → 128x7x17
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2),  # 128x7x17 → 64x18x38
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # 64x18x38 → 32x39x79
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2),  # 32x39x79 → 3x80x160
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, z):
        # z: (batch, 95)
        h = self.fc(z)  # (batch, 3584)
        h = h.view(-1, 256, 2, 7)  # (batch, 256, 2, 7)
        
        recon = self.deconv_layers(h)
        return recon
```

### VAE Class

File: `autoencoder_rgb/vae_rgb.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from encoder_rgb import VariationalEncoderRGB
from decoder_rgb import DecoderRGB

class VAE(nn.Module):
    def __init__(self, latent_dim=95, lr=1e-4, kl_beta=1e-6):
        super(VAE, self).__init__()
        
        self.encoder = VariationalEncoderRGB(latent_dim)
        self.decoder = DecoderRGB(latent_dim)
        
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )
        
        self.mse_loss = nn.MSELoss()
        self.kl_beta = kl_beta
        self.latent_dim = latent_dim
    
    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def compute_loss(self, x, recon, mu, logvar):
        # Reconstruction loss
        recon_loss = self.mse_loss(recon, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_step(self, dataloader, device):
        self.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        for batch in dataloader:
            x = batch.to(device)
            
            self.optimizer.zero_grad()
            recon, mu, logvar = self.forward(x)
            loss, recon_loss, kl_loss = self.compute_loss(x, recon, mu, logvar)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        return {
            'loss': total_loss / len(dataloader),
            'recon_loss': total_recon_loss / len(dataloader),
            'kl_loss': total_kl_loss / len(dataloader)
        }
    
    def encode(self, x):
        self.encoder.eval()
        with torch.no_grad():
            z, mu, logvar = self.encoder(x)
        return z
    
    def decode(self, z):
        self.decoder.eval()
        with torch.no_grad():
            recon = self.decoder(z)
        return recon
    
    def save(self, path):
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'latent_dim': self.latent_dim
        }, path)
    
    def load(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Training VAE

### Dataset Preparation

File: `autoencoder_rgb/vae_rgb.py`

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob

class RGBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png'))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Data transforms
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
])

dataset = RGBDataset(root_dir='RGB_DATA_COLLECTION/dataset_new_16000', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### Training Loop

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize VAE
vae = VAE(latent_dim=95, lr=1e-4, kl_beta=1e-6).to(device)

# TensorBoard writer
writer = SummaryWriter(log_dir='runs/vae_rgb_32k_from_scratch_backup')

# Training parameters
num_epochs = 100
checkpoint_frequency = 10

for epoch in range(num_epochs):
    metrics = vae.train_step(dataloader, device)
    
    # Log to TensorBoard
    writer.add_scalar('Loss/Total', metrics['loss'], epoch)
    writer.add_scalar('Loss/Reconstruction', metrics['recon_loss'], epoch)
    writer.add_scalar('Loss/KL', metrics['kl_loss'], epoch)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"  Total Loss: {metrics['loss']:.4f}")
    print(f"  Recon Loss: {metrics['recon_loss']:.4f}")
    print(f"  KL Loss: {metrics['kl_loss']:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % checkpoint_frequency == 0:
        vae.save(f'autoencoder_rgb/model/vae_rgb_epoch_{epoch+1}.pth')
        print(f"  Saved checkpoint at epoch {epoch+1}")

writer.close()
```

## Sử dụng Encoder trong PPO

File: `encoder_init_rgb_v2.py`

```python
import torch
import numpy as np
from autoencoder_rgb.encoder_rgb import VariationalEncoderRGB

class EncodeStateRGBV2:
    def __init__(self, latent_dim=95, device='cuda'):
        self.device = device
        self.latent_dim = latent_dim
        
        # Load encoder
        self.encoder = VariationalEncoderRGB(latent_dim).to(device)
        self.encoder.eval()
    
    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'encoder_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint)
        print(f"Loaded encoder weights from {checkpoint_path}")
    
    def process_observation(self, rgb_image):
        """
        Process RGB image to latent vector.
        
        Args:
            rgb_image: numpy array (80, 160, 3) or (batch, 80, 160, 3)
        
        Returns:
            latent: numpy array (95,) or (batch, 95)
        """
        # Ensure correct shape
        if len(rgb_image.shape) == 3:
            rgb_image = rgb_image.unsqueeze(0)  # Add batch dimension
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        if isinstance(rgb_image, np.ndarray):
            rgb_tensor = torch.from_numpy(rgb_image).permute(0, 3, 1, 2).float()
        else:
            rgb_tensor = rgb_image.permute(0, 3, 1, 2)
        
        # Normalize to [0, 1] if needed
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        
        # Move to device
        rgb_tensor = rgb_tensor.to(self.device)
        
        # Encode
        with torch.no_grad():
            latent, mu, logvar = self.encoder(rgb_tensor)
        
        # Return as numpy
        return latent.cpu().numpy().squeeze(0)

# Usage in PPO training
encoder_state = EncodeStateRGBV2(latent_dim=95, device='cuda')
encoder_state.load_weights('autoencoder_rgb/model/vae_rgb_epoch_100.pth')

# Process observation during training
rgb_image, navigation_state = env.get_observation()
latent = encoder_state.process_observation(rgb_image)
full_state = np.concatenate([latent, navigation_state])  # 95 + 5 = 100 dims
```

## Loss Function

### Reconstruction Loss (MSE)
```python
MSE_loss = 1/N * Σ(x - x_recon)²
```

### KL Divergence Loss
```python
KL_loss = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

### Total Loss
```python
total_loss = MSE_loss + KL_BETA * KL_loss
```

Trong đó `KL_BETA` là hyperparameter điều chỉnh trade-off giữa reconstruction quality và latent space regularization.

## Hyperparameters

| Hyperparameter | Giá trị | Mô tả |
|----------------|---------|-------|
| `latent_dim` | 95 | S chiều latent space |
| `learning_rate` | 1e-4 | Learning rate cho Adam optimizer |
| `kl_beta` | 1e-6 | Weight cho KL loss |
| `batch_size` | 32 | Batch size cho training |
| `num_epochs` | 100 | Số epochs training |

## Visualize Results

### Reconstruct Images

File: `autoencoder_rgb/reconstructor_rgb.py`

```python
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def visualize_reconstruction(vae, image_path, device):
    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Reconstruct
    with torch.no_grad():
        z, mu, logvar = vae.encoder(img_tensor)
        recon = vae.decoder(z)
    
    # Convert to display format
    original = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    reconstructed = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed Image')
    plt.tight_layout()
    plt.savefig('autoencoder_rgb/reconstructed/comparison.png')
    plt.show()
```

### Latent Space Visualization (t-SNE)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_latent_space(vae, dataloader, device):
    # Collect latent vectors
    latents = []
    labels = []  # Optional: category labels
    
    vae.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z, mu, logvar = vae.encoder(batch)
            latents.append(z.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    latents_2d = tsne.fit_transform(latents)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.5)
    plt.title('Latent Space Visualization (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('autoencoder_rgb/reconstructed/latent_space.png')
    plt.show()
```

## Troubleshooting

### Lỗi: "Reconstruction quá mờ"
- Tăng KL_BETA để regularization mạnh hơn
- Tăng số lượng training epochs
- Kiểm tra data quality

### Lỗi: "Latent space không hữu ích cho PPO"
- Thử latent_dim lớn hơn (128, 256)
- Adjust KL_BETA nhỏ hơn
- Train với dataset đa dạng hơn

### Lỗi: "CUDA Out of Memory"
- Giảm batch_size
- Giảm độ phân giải nh đầu vào
- Dùng gradient accumulation

## Next Steps

- [09_PPO.md](09_PPO.md) - PPO training
- [10_Dataset.md](10_Dataset.md) - Dataset collection
- [13_Testing.md](13_Testing.md) - Testing VAE quality
