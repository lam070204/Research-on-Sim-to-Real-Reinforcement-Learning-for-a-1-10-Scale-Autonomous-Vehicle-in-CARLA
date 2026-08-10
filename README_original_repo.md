# Hệ thống học tăng cường cho xe tự hành tỉ lệ 1/10

Dự án sử dụng RoadRunner, CARLA, ảnh RGB, Variational Autoencoder (VAE) và Proximal Policy Optimization (PPO) để huấn luyện xe tự hành chạy trên sa bàn cố định, hướng đến triển khai trên NVIDIA Jetson Orin và xe HSP 94123 tỉ lệ 1/10.

## Kiến trúc

```text
RoadRunner/OpenDRIVE
        ↓
CARLA map
        ↓
RGB camera 160×80
        ↓
VAE encoder 95 chiều
        ↓
Ghép navigation 5 chiều
        ↓
PPO state 100 chiều
        ↓
Steering + throttle
```
## Mô hình triển khai
RGB 160×80×3
       ↓
      VAE
       ↓
     95 z
       +
 5 trạng thái xe
       ↓
  STATE 100
       ↓
      PPO
   ┌───┴────┐
   ↓        ↓
 Actor    Critic
   ↓        ↓
2 action   V(s)
   ↓
steer + throttle
   ↓
    CARLA
   ↓
xe chạy
   ↓
reward
   ↓
PPO dùng reward để sửa Actor/Critic

## Pipeline Semantic

- `continuous_driver.py`
- `encoder_init.py`
- `simulation/environment.py`
- `simulation/sensors.py`

## Pipeline RGB

- `continuous_driver_rgb.py`
- `encoder_init_rgb.py`
- `simulation/environment_rgb.py`
- `autoencoder_rgb/encoder_rgb.py`
- `autoencoder_rgb/decoder_rgb.py`
- `autoencoder_rgb/vae_rgb.py`
- `autoencoder_rgb/reconstructor_rgb.py`

## Tài liệu

- [Tổng quan](docs/01_Project_Overview.md)
- [Cấu trúc project](docs/02_Project_Structure.md)
- [Cài đặt](docs/03_Installation.md)
- [RoadRunner và map](docs/04_RoadRunner_Map.md)
- [CARLA](docs/05_CARLA.md)
- [Sensors](docs/06_Sensors.md)
- [Environment](docs/07_Environment.md)
- [VAE RGB](docs/08_VAE_RGB.md)
- [PPO](docs/09_PPO.md)
- [Dataset](docs/10_Dataset.md)
- [Reward và điều khiển](docs/11_Reward_Control.md)
- [Checkpoint](docs/12_Checkpoint.md)
- [Testing](docs/13_Testing.md)
- [Deployment](docs/14_Deployment.md)
- [Troubleshooting](docs/15_Troubleshooting.md)
- [API](docs/16_API.md)
- [Changelog](docs/CHANGELOG.md)

## Train VAE RGB

```powershell
python .\autoencoder_rgb\vae_rgb.py
```

## Train PPO RGB

```powershell
python .\continuous_driver_rgb.py `
  --exp-name ppo `
  --train true `
  --town mapden `
  --load-checkpoint false `
  --total-timesteps 500000 `
  --episode-length 3000 `
  --collect-rgb-dataset false
```

## Test PPO RGB

```powershell
python .\continuous_driver_rgb.py `
  --exp-name ppo `
  --train false `
  --town mapden `
  --load-checkpoint true `
  --test-timesteps 100000 `
  --episode-length 3000 `
  --collect-rgb-dataset false
```
