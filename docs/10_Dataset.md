# 10 Dataset

## Dataset Collection

Tài liệu về việc thu thập và quản lý dataset cho training VAE và PPO.

## Tổng quan

Dự án sử dụng hai loại dataset chính:

1. **RGB Dataset**: Ảnh RGB từ camera để training VAE
2. **Experience Dataset**: Experience tuples cho off-policy RL (nếu dùng)

## RGB Dataset cho VAE

### Cấu trúc Dataset

```
RGB_DATA_COLLECTION/
├── README.txt
├── dataset_new_16000/
│   ├── image_00001.png
│   ├── image_00002.png
│   ├── ...
│   └── image_016000.png
└── scripts/
    ├── collect_rgb_autopilot.py
    └── ...
```

### Thu thập Dataset với Autopilot

File: `collect_rgb_autopilot.py`

```python
import carla
import os
import time
import numpy as np
from PIL import Image

class DatasetCollector:
    def __init__(self, town='Town07', output_dir='RGB_DATA_COLLECTION/dataset_new_16000'):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.vehicle = None
        self.camera = None
        self.image_count = 0
        
    def spawn_vehicle_with_camera(self):
        # Spawn vehicle
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.lincoln*')[0]
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Attach RGB camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '160')
        camera_bp.set_attribute('image_size_y', '80')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(
            carla.Location(x=2.5, y=0.0, z=1.5),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        self.camera.listen(lambda image: self.save_image(image))
    
    def save_image(self, image):
        # Convert image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb_array = array[:, :, :3][:, :, ::-1]  # BGRA to RGB
        
        # Save as PNG
        filename = os.path.join(
            self.output_dir, 
            f'image_{self.image_count:05d}.png'
        )
        
        pil_image = Image.fromarray(rgb_array)
        pil_image.save(filename)
        
        self.image_count += 1
        
        if self.image_count % 100 == 0:
            print(f"Collected {self.image_count} images")
    
    def set_autopilot(self, enabled=True):
        self.vehicle.set_autopilot(enabled)
    
    def run(self, num_images=16000, duration_per_spawn=60):
        print(f"Starting dataset collection. Target: {num_images} images")
        
        while self.image_count < num_images:
            # Spawn vehicle with camera
            self.spawn_vehicle_with_camera()
            self.set_autopilot(True)
            
            # Wait for duration
            start_time = time.time()
            while time.time() - start_time < duration_per_spawn:
                time.sleep(0.1)
                
                if self.image_count >= num_images:
                    break
            
            # Cleanup
            self.set_autopilot(False)
            self.camera.stop()
            self.camera.destroy()
            self.vehicle.destroy()
            
            print(f"Spawn complete. Total images: {self.image_count}")
        
        print(f"Dataset collection complete. Total: {self.image_count} images")
        self.client.reload_world()

if __name__ == '__main__':
    collector = DatasetCollector(
        town='Town07',
        output_dir='RGB_DATA_COLLECTION/dataset_new_16000'
    )
    collector.run(num_images=16000, duration_per_spawn=60)
```

### Dataset Statistics

| Thống kê | Giá trị |
|----------|---------|
| Tổng số ảnh | 16,000 |
| Độ phân giải | 160x80 pixels |
| Định dạng | PNG (RGB) |
| Dung lượng | ~50 MB |
| Maps | Town07 |
| Vehicles | Lincoln MKZ 2017 |
| Conditions | Clear weather, daytime |

### Data Augmentation

```python
from torchvision import transforms

# Augmentation transforms for VAE training
train_transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
])

# No augmentation for validation
val_transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])
```

### Dataset Loader

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob

class RGBDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Create data loaders
train_dataset = RGBDataset(
    root_dir='RGB_DATA_COLLECTION/dataset_new_16000',
    transform=train_transform
)

val_dataset = RGBDataset(
    root_dir='RGB_DATA_COLLECTION/dataset_new_16000',
    transform=val_transform,
    max_samples=1000  # Use 1000 for validation
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)
```

## Experience Dataset cho Off-Policy RL

### Experience Buffer

```python
from collections import deque
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store experience tuple.
        
        Args:
            state: tuple (latent, navigation)
            action: numpy array (2,)
            reward: float
            next_state: tuple (latent, navigation)
            done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
```

### Prioritized Experience Replay

```python
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        # Store with max priority (new experiences are important)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Get batch
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights),
            indices
        )
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
```

## Dataset Validation

### Check Data Quality

```python
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def validate_dataset(dataset_dir, num_samples=10):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png'))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i >= len(image_paths):
            break
        
        img = Image.open(image_paths[i])
        ax.imshow(img)
        ax.set_title(f'{os.path.basename(image_paths[i])}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_validation.png')
    plt.show()
    
    # Statistics
    images = []
    for path in image_paths[:100]:
        img = np.array(Image.open(path))
        images.append(img)
    
    images = np.array(images)
    
    print(f"Dataset shape: {images.shape}")
    print(f"Mean pixel value: {images.mean():.2f}")
    print(f"Std pixel value: {images.std():.2f}")
    print(f"Min pixel value: {images.min()}")
    print(f"Max pixel value: {images.max()}")
```

### Check for Duplicates

```python
import hashlib

def find_duplicates(dataset_dir):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png'))
    hash_dict = {}
    duplicates = []
    
    for path in image_paths:
        with open(path, 'rb') as f:
            img_hash = hashlib.md5(f.read()).hexdigest()
        
        if img_hash in hash_dict:
            duplicates.append((path, hash_dict[img_hash]))
        else:
            hash_dict[img_hash] = path
    
    print(f"Found {len(duplicates)} duplicate images")
    return duplicates
```

## Dataset Management

### Split Dataset

```python
from sklearn.model_selection import train_test_split
import shutil

def split_dataset(dataset_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png'))
    
    train_paths, temp_paths = train_test_split(
        image_paths, 
        test_size=(1 - train_ratio),
        random_state=42
    )
    
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42
    )
    
    # Create directories
    os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'test'), exist_ok=True)
    
    # Copy files
    for path in train_paths:
        shutil.copy(path, os.path.join(dataset_dir, 'train', os.path.basename(path)))
    
    for path in val_paths:
        shutil.copy(path, os.path.join(dataset_dir, 'val', os.path.basename(path)))
    
    for path in test_paths:
        shutil.copy(path, os.path.join(dataset_dir, 'test', os.path.basename(path)))
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
```

### Clean Dataset

```python
def clean_dataset(dataset_dir):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png'))
    
    for path in image_paths:
        try:
            img = Image.open(path)
            img.verify()  # Verify it's a valid image
            
            # Re-open to check size
            img = Image.open(path)
            if img.size != (160, 80):
                print(f"Removing {path} - wrong size: {img.size}")
                os.remove(path)
        
        except Exception as e:
            print(f"Removing {path} - error: {e}")
            os.remove(path)
    
    print("Dataset cleaning complete")
```

## Best Practices

### Data Collection
- Thu thập đa dạng scenarios (straight, curve, intersection)
- Đảm bảo chất lượng ảnh tốt (không blur, không artifacts)
- Thu thập  nhiều thời điểm khác nhau trong simulation

### Data Storage
- Dùng định dạng PNG cho lossless compression
- Organize theo thư mục có cấu trúc
- Backup dataset thường xuyên

### Data Quality
- Validate ảnh ngay sau khi thu thập
- Remove corrupted/duplicate images
- Check distribution của dataset

## Next Steps

- [08_VAE_RGB.md](08_VAE_RGB.md) - VAE training với dataset
- [09_PPO.md](09_PPO.md) - PPO training
- [14_Deployment.md](14_Deployment.md) - Deploy với trained model
