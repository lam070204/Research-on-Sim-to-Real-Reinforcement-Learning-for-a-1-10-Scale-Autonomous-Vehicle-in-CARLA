# Hệ thống học tăng cường cho xe tự hành tỉ lệ 1/10

Dự án sử dụng RoadRunner, CARLA, camera RGB, Variational Autoencoder (VAE) và Proximal Policy Optimization (PPO) để huấn luyện xe tự hành chạy trên sa bàn cố định, hướng đến triển khai trên NVIDIA Jetson Orin và xe HSP 94123 tỉ lệ 1/10.

## Kiến trúc tổng thể

```text
RoadRunner / OpenDRIVE
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

## Tài liệu chi tiết

1. [Tổng quan dự án](docs/01_Project_Overview.md)
2. [Cấu trúc thư mục và file](docs/02_Project_Structure.md)
3. [Cài đặt](docs/03_Installation.md)
4. [RoadRunner và map](docs/04_RoadRunner_Map.md)
5. [CARLA](docs/05_CARLA.md)
6. [Sensors](docs/06_Sensors.md)
7. [Environment](docs/07_Environment.md)
8. [VAE RGB](docs/08_VAE_RGB.md)
9. [PPO](docs/09_PPO.md)
10. [Dataset](docs/10_Dataset.md)
11. [Reward và điều khiển](docs/11_Reward_Control.md)
12. [Checkpoint](docs/12_Checkpoint.md)
13. [Testing](docs/13_Testing.md)
14. [Deployment](docs/14_Deployment.md)
15. [Troubleshooting](docs/15_Troubleshooting.md)
16. [API](docs/16_API.md)
17. [Hướng dẫn kế thừa](docs/17_Developer_Guide.md)
18. [Changelog](docs/CHANGELOG.md)

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
