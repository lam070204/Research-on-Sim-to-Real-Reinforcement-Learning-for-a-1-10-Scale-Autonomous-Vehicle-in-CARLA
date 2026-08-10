# 02 Project Structure

## Cấu trúc Thư mục

```
Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/
│
├── docs/                           # Documentation chi tiết
│   ├── 01_Project_Overview.md      # Tổng quan dự án
│   ├── 02_Project_Structure.md     # Cấu trúc thư mục (file này)
│   ├── 03_Installation.md          # Hướng dẫn cài đặt
│   ├── 04_RoadRunner_Map.md        # Tạo map với RoadRunner
│   ├── 05_CARLA.md                 # CARLA Simulator
│   ├── 06_Sensors.md               # Hệ thống sensor
│   ├── 07_Environment.md           # Environment CARLA
│   ├── 08_VAE_RGB.md               # VAE cho RGB
│   ├── 09_PPO.md                   # Thuật toán PPO
│   ├── 10_Dataset.md               # Dataset
│   ├── 11_Reward_Control.md        # Reward function
│   ├── 12_Checkpoint.md            # Checkpoint system
│   ├── 13_Testing.md               # Testing
│   ├── 14_Deployment.md            # Deployment
│   ├── 15_Troubleshooting.md       # Troubleshooting
│   ├── 16_API.md                   # API Reference
│   ├── 17_Developer_Guide.md       # Developer Guide
│   └── CHANGELOG.md                # Changelog
│
├── autoencoder/                    # Autoencoder cũ (semantic)
│   ├── decoder.py
│   ├── encoder.py
│   ├── reconstructor.py
│   └── vae.py
│
├── autoencoder_rgb/                # Autoencoder cho RGB
│   ├── __init__.py
│   ├── encoder_rgb.py              # Encoder network
│   ├── decoder_rgb.py              # Decoder network
│   ├── vae_rgb.py                  # Training script
│   ├── reconstructor_rgb.py        # Reconstruct ảnh
│   ├── model/                      # Model checkpoints
│   └── reconstructed/              # Ảnh reconstructed
│
├── autoencoder-semantic/           # Autoencoder semantic segmentation
│   ├── encoder_rgb.py
│   ├── decoder_rgb.py
│   ├── vae_rgb.py
│   ├── dataset/
│   ├── model/
│   └── reconstructed/
│
├── networks/                       # Neural Networks
│   ├── on_policy/
│   │   └── ppo/
│   │       ├── agent.py            # PPO Agent
│   │       ├── agent_v2.py         # PPO Agent V2
│   │       └── ppo.py              # Policy networks (Actor-Critic)
│   └── off_policy/
│       └── ddqn/
│           └── ...
│
├── simulation/                     # CARLA Simulation
│   ├── connection.py               # CARLA connection wrapper
│   ├── environment.py              # Base environment
│   ├── environment_rgb.py          # RGB environment (old)
│   ├── environment_rgb_v2.py       # RGB environment V2 (chính)
│   ├── sensors.py                  # Sensor definitions
│   └── settings.py                 # Configuration settings
│
├── checkpoints/                    # Training checkpoints
│   ├── DDQN/
│   └── PPO/
│       └── Town07/                 # Checkpoints cho Town07
│           ├── ppo_policy_0_.pth
│           ├── ppo_policy_1_.pth
│           └── ...
│
├── preTrained_models/              # Pre-trained models
│   ├── ddqn/
│   └── ppo/
│
├── runs/                           # TensorBoard logs
│   ├── PPO_0.2_1000/
│   ├── PPO_0.2_5000/
│   ├── PPO_0.2_10000/
│   ├── PPO_0.2_100000/
│   ├── PPO_0.2_200000/
│   ├── PPO_0.2_300000/
│   ├── PPO_0.2_500000/
│   ├── PPO_0.2_600000/
│   ├── PPO_0.2_650000/
│   ├── PPO_0.2_1000000/
│   ├── PPO_0.2_2000000_TEST/
│   └── vae_rgb_*/
│
├── RGB_DATA_COLLECTION/            # Dataset RGB
│   ├── README.txt
│   ├── dataset_new_16000/
│   └── scripts/
│
├── carla/                          # CARLA Python API
│   └── carla-0.9.13-py3.8-win-amd64.egg
│
├── poetry/                         # Poetry configuration
│   ├── pyproject.toml
│   └── poetry.lock
│
├── continuous_driver_rgb_v2.py     # Main training script (RGB V2)
├── continuous_driver_rgb.py        # Main training script (RGB old)
├── continuous_driver.py            # Main training script (base)
├── discrete_driver.py              # Discrete action driver
├── encoder_init_rgb_v2.py          # Encoder initialization V2
├── encoder_init_rgb.py             # Encoder initialization
├── encoder_init.py                 # Encoder initialization (base)
├── collect_rgb_autopilot.py        # Thu thập dataset RGB
├── measure_map.py                  # Đo đạc map
├── parameters.py                   # Global parameters
├── requirements.txt                # Python dependencies
├── README.md                       # README chính
├── README_original_repo.md         # README từ repo gốc
└── LICENSE.md                      # License
```

## Mô tả Chi tiết Các Thư mục

### `autoencoder_rgb/`
Chứa implementation của Variational Autoencoder cho dữ liệu RGB:
- **encoder_rgb.py**: Mạng encoder với 4 lớp Conv2D, output 95-dim latent space
- **decoder_rgb.py**: Mạng decoder với 4 lớp ConvTranspose2D
- **vae_rgb.py**: Script training VAE, kết hợp encoder và decoder
- **model/**: Lưu trữ model checkpoints sau training

### `networks/on_policy/ppo/`
Chứa implementation của thuật toán PPO:
- **agent_v2.py**: Class `PPOAgent` với buffer, get_action, learn, save/load
- **ppo.py**: Class `ActorCritic` định nghĩa policy network và value network

### `simulation/`
Chứa các class tương tác với CARLA:
- **connection.py**: Wrapper kết nối CARLA client
- **environment_rgb_v2.py**: Class `CarlaEnvironmentRGB` - environment chính cho training
- **sensors.py**: Định nghĩa các sensor (Camera, Collision, IMU, GNSS)
- **settings.py**: Configuration constants (FPS, camera params, reward factors)

### `checkpoints/PPO/`
Lưu trữ các checkpoint của PPO trong quá trình training:
- Mỗi town có thư mục riêng (Town07, Town05, ...)
- Checkpoint được lưu sau mỗi N steps (checkpoint_frequency)
- Định dạng: `ppo_policy_{step}_.pth`

### `runs/`
Chứa logs TensorBoard cho các experiment:
- TensorBoard command: `tensorboard --logdir=runs/`
- Theo dõi: loss, reward, entropy, value loss

## File Cấu hình Chính

| File | Mục đích |
|------|----------|
| `parameters.py` | Global parameters (learning rate, gamma, latent dim, etc.) |
| `simulation/settings.py` | Environment settings (reward factors, spawn points, camera params) |
| `requirements.txt` | Python dependencies |
| `poetry/pyproject.toml` | Poetry project configuration |

## File Thực thi Chính

| File | Mục đích |
|------|----------|
| `continuous_driver_rgb_v2.py` | Training/testing PPO với RGB camera |
| `collect_rgb_autopilot.py` | Thu thập dataset RGB từ autopilot |
| `autoencoder_rgb/vae_rgb.py` | Training VAE |
| `autoencoder_rgb/reconstructor_rgb.py` | Reconstruct ảnh từ latent |
