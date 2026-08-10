# Autonomous Driving in CARLA using Deep Reinforcement Learning

Dự án nghiên cứu và phát triển hệ thống lái tự động trong môi trường mô phỏng CARLA sử dụng Deep Reinforcement Learning (PPO) và Variational Autoencoder (VAE) để xử lý dữ liệu camera RGB.

---

## 📋 Mục Lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Phương thức hoạt động](#phương-thức-hoạt-động)
  - [1. Camera RGB và Xử lý ảnh](#1-camera-rgb-và-xử-lý-ảnh)
  - [2. Variational Autoencoder (VAE)](#2-variational-autoencoder-vae)
  - [3. Navigation State](#3-navigation-state)
  - [4. PPO Agent](#4-ppo-agent)
  - [5. Environment và Reward](#5-environment-và-reward)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Sử dụng](#sử-dụng)
- [Documentation chi tiết](#documentation-chi tiết)

---

## 📋 Tổng quan

Dự án này triển khai một hệ thống lái tự động hoàn chỉnh cho xe tự hành trong môi trường CARLA, sử dụng:

- **PPO (Proximal Policy Optimization)**: Thuật toán Deep Reinforcement Learning on-policy
- **VAE (Variational Autoencoder)**: Mạng neural tự mã hóa để nén ảnh RGB thành latent space
- **CARLA Simulator**: Môi trường mô phỏng lái xe 3D mã nguồn mở

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        CARLA Environment                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ RGB Camera  │  │ Collision    │  │ Navigation (state)   │   │
│  │ 160x80x3    │  │ Sensor       │  │ - throttle           │   │
│  │ @ 20 FPS    │  │              │  │ - speed              │   │
│  └──────┬──────┘  └──────────────┘  │ - steer              │   │
│         │                           │ - lateral distance   │   │
│         │                           │ - heading error      │   │
│         ▼                           └─────────────────────┘   │
│  ┌─────────────────┐                                          │
│  │ VAE Encoder     │                                          │
│  │ (95-dim latent) │                                          │
│  └────────┬────────┘                                          │
│           │                                                    │
│           ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              PPO Agent (Policy Network)                 │   │
│  │  Input: latent (95) + navigation (5) = 100 dimensions   │   │
│  │  Output: continuous action (steer, throttle)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Phương thức hoạt động

### 1. Camera RGB và Xử lý ảnh

#### **Thông số camera** (`simulation/sensors.py`)

| Thông số | Giá trị |
|----------|---------|
| Độ phân giải | 160x80 pixels |
| FPS | 20 |
| FOV | 90° |
| Vị trí X | 2.5m (trước xe) |
| Vị trí Y | 0.0m (giữa xe) |
| Vị trí Z | 1.5m (độ cao) |
| Pitch | 0.0° |
| Yaw | 0.0° |
| Roll | 0.0° |

#### **Chức năng các file liên quan:**

| File | Chức năng |
|------|-----------|
| `simulation/sensors.py` | Định nghĩa thông số camera và class `CameraSensorRGBPPO` để thu nhận ảnh RGB từ CARLA |
| `simulation/environment_rgb_v2.py` | Class `CarlaEnvironmentRGB` quản lý môi trường, spawn xe, và thu thập dữ liệu camera |
| `continuous_driver_rgb_v2.py` | Script chính để train/test PPO với dữ liệu RGB |

#### **Quy trình xử lý:**

1. Camera trong CARLA capture ảnh RGB 160x80x3
2. Ảnh được chuyển đổi từ định dạng CARLA (BGRA) sang RGB
3. Ảnh được normalize về [0, 1] bằng cách chia cho 255
4. Đưa vào VAE Encoder để trích xuất latent vector

---

### 2. Variational Autoencoder (VAE)

#### **Mục đích:**
Nén ảnh RGB 160x80x3 (38,400 giá trị pixel) thành latent vector 95 chiều, giảm đáng kể chiều dữ liệu ầu vào cho PPO.

#### **Kiến trúc VAE** (`autoencoder_rgb/vae_rgb.py`):

```
Encoder:
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
  z = μ + σ * ε (reparameterization trick)
  ↓
  Output: 95-dim latent vector

Decoder:
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

#### **Các file quan trọng:**

| File | Chức năng |
|------|-----------|
| `autoencoder_rgb/encoder_rgb.py` | Định nghĩa kiến trúc `VariationalEncoderRGB` |
| `autoencoder_rgb/decoder_rgb.py` | Định nghĩa kiến trúc `DecoderRGB` |
| `autoencoder_rgb/vae_rgb.py` | Training script cho VAE, kết hợp encoder và decoder |
| `autoencoder_rgb/reconstructor_rgb.py` | Script để reconstruct ảnh từ latent space |
| `encoder_init_rgb_v2.py` | Class `EncodeStateRGBV2` để load encoder và process observation |

#### **Loss function:**
```python
total_loss = MSE_loss(reconstructed, original) + KL_BETA * KL_divergence
```

---

### 3. Navigation State

#### **5 thành phần navigation** (`simulation/environment_rgb_v2.py`):

| Index | Thành phần | Mô tả | Range |
|-------|------------|-------|-------|
| 0 | `throttle` | Throttle thực tế đã áp dụng | 0.0 - 1.0 |
| 1 | `normalized_speed` | Tốc độ / target_speed | 0.0 - 1.5 |
| 2 | `previous_steer` | Góc lái trước đó | -1.0 - 1.0 |
| 3 | `normalized_lateral` | Khoảng cách ngang có dấu / lane_limit | -1.0 - 1.0 |
| 4 | `normalized_heading` | Sai lệch hướng có dấu / heading_limit | -1.0 - 1.0 |

#### **Công thức tính:**

```python
# Signed lateral distance (dương = bên trái route, âm = bên phải)
signed_distance = cross_product(route_direction, vehicle_position) / route_length

# Signed heading error
heading_error = atan2(vehicle_fwd) - atan2(waypoint_fwd)

# Normalized
normalized_lateral = clip(signed_distance / 3.0, -1.0, 1.0)
normalized_heading = clip(heading_error / 20°, -1.0, 1.0)
```

#### **State vector hoàn chỉnh:**
```
state = [latent_rgb (95 dims), navigation (5 dims)] = 100 dimensions
```

---

### 4. PPO Agent

#### **Kiến trúc Policy Network:**

```
Input: 100 dimensions (95 latent + 5 navigation)
↓
Fully Connected (256) + Tanh
↓
Fully Connected (256) + Tanh
↓
┌──────────────────┬──────────────────┐
│   Actor Head     │   Critic Head    │
│ (policy network) │  (value network) │
│                  │                  │
│ Mean: FC(256, 2) │   FC(256, 1)     │
│ Log std: (2,)    │   Output: V(s)   │
│ Output: μ, σ     │                  │
└──────────────────┴──────────────────┘
```

#### **Action space:**
| Action | Ý nghĩa | Range |
|--------|---------|-------|
| 0 | Steering | -1.0 - 1.0 |
| 1 | Throttle | 0.0 - 1.0 (sau khi transform từ [-1, 1]) |

#### **Các file liên quan:**

| File | Chức năng |
|------|-----------|
| `networks/on_policy/ppo/agent.py` | Class `PPOAgent` với bộ nhớ và hàm học |
| `networks/on_policy/ppo/policy.py` | Định nghĩa actor và critic networks |
| `continuous_driver_rgb_v2.py` | Training loop chính cho PPO |

#### **PPO Loss:**
```python
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
L_VF = (V(s) - V_target)²
L_ENTROPY = -H(π)
Total Loss = -L_CLIP + c1 * L_VF - c2 * L_ENTROPY
```

---

### 5. Environment và Reward

#### **Reward function** (`simulation/environment_rgb_v2.py`):

```python
# Factors
centering_factor = max(1.0 - distance_from_center / 3.0, 0.0)
angle_factor = max(1.0 - abs(heading_error) / 20°, 0.0)
speed_factor = velocity / target_speed (nếu < min_speed)
             = 1.0 - (velocity - target_speed) / (max_speed - target_speed) (nếu > target_speed)

# Base reward
reward = 1.2 * speed_factor * centering_factor * angle_factor

# Heading bonus
if abs(heading_error) < 5°: reward += 0.4
elif abs(heading_error) < 10°: reward += 0.2

# Lane deviation penalties
if distance_from_center > 2.5m: reward -= 2.0
elif distance_from_center > 2.0m: reward -= 1.0
elif distance_from_center > 1.5m: reward -= 0.5

# Terminal rewards
if collision: reward = -10
if off_road: reward = -10
if stuck: reward = -10
if extreme_overspeed: reward = -5
if route_completed: reward += 10.0
```

#### **Điều kiện kết thúc episode:**

| Điều kiện | Reward | Mô tả |
|-----------|--------|-------|
| Collision | -10 | Xe va chạm vật thể |
| Off road | -10 | Xe ra khỏi lane > 3m |
| Stuck | -10 | Xe đứng yên > 10s sau spawn |
| Extreme overspeed | -5 | Tốc độ > 33 km/h |
| Route completed | +10 | Đi hết route |
| Time limit | 0 | Đạt 50,000 steps |

#### **Safe Spawn System:**
- Sử dụng 8 spawn points an toàn: [1, 4, 6, 7, 8, 9, 10, 11]
- Lần lượt dùng từng spawn, sau đó shuffle và lặp lại
- Mỗi spawn có route riêng được build động

---

## 📁 Cấu trúc thư mục

```
Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/
├── autoencoder/                  # Autoencoder cũ (semantic)
├── autoencoder_rgb/              # Autoencoder cho RGB
│   ├── encoder_rgb.py           # Encoder network
│   ├── decoder_rgb.py           # Decoder network
│   ├── vae_rgb.py               # Training script
│   ├── reconstructor_rgb.py     # Reconstruct ảnh
│   ├── model/                   # Model checkpoints
│   └── reconstructed/           # Ảnh reconstructed
├── autoencoder-semantic/         # Autoencoder semantic segmentation
├── networks/
│   ├── on_policy/
│   │   └── ppo/
│   │       ├── agent.py         # PPO Agent
│   │       └── policy.py        # Policy networks
│   └── off_policy/
├── simulation/
│   ├── connection.py            # CARLA connection
│   ├── environment.py           # Base environment
│   ├── environment_rgb.py       # RGB environment (old)
│   ├── environment_rgb_v2.py    # RGB environment V2
│   ├── sensors.py               # Sensor definitions
│   └── settings.py              # Configuration
├── checkpoints/PPO/             # PPO checkpoints
├── preTrained_models/           # Pre-trained models
├── runs/                        # TensorBoard training logs
│   ├── PPO_0.2_1000/            # Log PPO với 1000 steps
│   ├── PPO_0.2_5000/            # Log PPO với 5000 steps
│   ├── PPO_0.2_10000/           # Log PPO với 10000 steps
│   ├── PPO_0.2_100000/          # Log PPO với 100k steps
│   ├── PPO_0.2_500000/          # Log PPO với 500k steps
│   │   ├── mapden/              # Log training trên map đơn giản
│   │   └── bandoxethuc/         # Log training trên bản đồ thực tế
│   ├── PPO_0.2_1000000/         # Log PPO với 1M steps
│   ├── PPO_0.2_2000000_TEST/    # Log PPO test với 2M steps
│   └── vae_rgb_*/               # Log training VAE cho RGB
├── docs/                        # Documentation chi tiết
├── continuous_driver_rgb_v2.py  # Main training script
├── encoder_init_rgb_v2.py       # Encoder initialization
├── discrete_driver.py           # Discrete action driver
├── parameters.py                # Global parameters
└── README.md                    # File này
```

---

## 🚀 Cài đặt

### Yêu cầu:
- Python 3.8+
- CARLA 0.9.13
- PyTorch 1.8+
- CUDA (khuyến khích)

### Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

### Cài đặt CARLA:
1. Tải CARLA 0.9.13 từ https://carla.org/
2. Thêm CARLA egg vào Python path:
   ```bash
   cp carla/carla-0.9.13-py3.8-win-amd64.egg <CARLA_ROOT>/PythonAPI/carla/dist/
   ```

---

## 🎯 Sử dụng

### Training VAE (RGB Autoencoder):
```bash
cd autoencoder_rgb
python vae_rgb.py
```

### Training PPO với RGB:
```bash
python continuous_driver_rgb_v2.py \
    --exp-name ppo \
    --town Town07 \
    --train true \
    --load-checkpoint false \
    --total-timesteps 1000000
```

### Test PPO:
```bash
python continuous_driver_rgb_v2.py \
    --exp-name ppo \
    --town Town07 \
    --train false \
    --load-checkpoint true \
    --test-timesteps 10000
```

### Thu thập dataset RGB:
```bash
python continuous_driver_rgb_v2.py \
    --exp-name ppo \
    --town Town07 \
    --train false \
    --load-checkpoint true \
    --collect-rgb-dataset true \
    --dataset-root RGB_DATA_COLLECTION/dataset_new_16000 \
    --dataset-images-per-spawn 2000
```

---

## 📚 Documentation chi tiết

| Tài liệu | Nội dung |
|----------|----------|
| [01_Project_Overview.md](docs/01_Project_Overview.md) | Tổng quan dự án |
| [02_Project_Structure.md](docs/02_Project_Structure.md) | Cấu trúc thư mục |
| [03_Installation.md](docs/03_Installation.md) | Hướng dẫn cài đặt |
| [06_Sensors.md](docs/06_Sensors.md) | Hệ thống sensor |
| [07_Environment.md](docs/07_Environment.md) | Environment CARLA |
| [08_VAE_RGB.md](docs/08_VAE_RGB.md) | VAE cho RGB |
| [09_PPO.md](docs/09_PPO.md) | Thuật toán PPO |
| [11_Reward_Control.md](docs/11_Reward_Control.md) | Reward function |

---

## 📄 License

MIT License - xem [LICENSE.md](LICENSE.md)

---

## 🔗 Repository gốc

https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning
