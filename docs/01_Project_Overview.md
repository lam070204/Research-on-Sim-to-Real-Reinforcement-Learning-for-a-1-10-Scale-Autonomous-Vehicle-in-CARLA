# 01 Project Overview

## Tổng quan Dự án

Dự án nghiên cứu và phát triển hệ thống lái tự động trong môi trường mô phỏng CARLA sử dụng **Deep Reinforcement Learning (PPO)** và **Variational Autoencoder (VAE)** để xử lý dữ liệu camera RGB.

## Mục tiêu

Xây dựng một agent có khả năng:
- Lái xe tự động trong môi trường CARLA
- Sử dụng camera RGB làm sensor chính
- Học các kỹ năng lái xe thông qua reinforcement learning
- Generalize tốt trên các map và tình huống khác nhau

## Công nghệ Chính

| Thành phần | Công nghệ | Mô tả |
|------------|-----------|-------|
| **Simulator** | CARLA 0.9.13 | Môi trường mô phỏng lái xe 3D mã nguồn mở |
| **RL Algorithm** | PPO (Proximal Policy Optimization) | Thuật toán Deep RL on-policy |
| **Vision Model** | VAE (Variational Autoencoder) | Nén ảnh RGB thành latent vector 95 chiều |
| **Deep Learning Framework** | PyTorch 1.8+ | Framework cho training và inference |
| **Language** | Python 3.8+ | Ngôn ngữ lập trình chính |

## Kiến trúc Hệ thống

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

## Luồng Dữ liệu

1. **Thu nhận ảnh**: Camera RGB trong CARLA capture ảnh 160x80x3 ở 20 FPS
2. **Tiền xử lý**: Chuyển đổi từ BGRA sang RGB, normalize về [0, 1]
3. **Encoding**: VAE Encoder nén ảnh thành latent vector 95 chiều
4. **State fusion**: Kết hợp latent vector với navigation state (5 chiều)
5. **Policy inference**: PPO Agent đưa ra action (steer, throttle)
6. **Thực thi action**: Action được áp dụng lên xe trong CARLA
7. **Nhận reward**: Environment tính reward dựa trên hiệu suất lái xe

## Các thành phần chính

### 1. Variational Autoencoder (VAE)
- **Encoder**: ConvNet 4 lớp, nén ảnh 38,400 pixel → 95-dim latent
- **Decoder**: ConvTranspose 4 lớp, reconstruct ảnh từ latent
- **Loss**: MSE Reconstruction + KL Divergence

### 2. PPO Agent
- **Actor Network**: 2 lớp Fully Connected (256 units), output mean và std
- **Critic Network**: 2 lớp Fully Connected (256 units), output value
- **Action Space**: Continuous (steering: -1→1, throttle: 0→1)

### 3. Environment
- **Observation Space**: 100 dimensions (95 latent + 5 navigation)
- **Reward Function**: Dựa trên tốc độ, vị trí lane, góc lệch heading
- **Terminal Conditions**: Collision, off-road, stuck, complete route

## Tài liệu Liên quan

- [02_Project_Structure.md](02_Project_Structure.md) - Cấu trúc thư mục
- [03_Installation.md](03_Installation.md) - Hướng dẫn cài đặt
- [06_Sensors.md](06_Sensors.md) - Hệ thống sensor
- [07_Environment.md](07_Environment.md) - Environment CARLA
- [08_VAE_RGB.md](08_VAE_RGB.md) - VAE cho RGB
- [09_PPO.md](09_PPO.md) - Thuật toán PPO
- [11_Reward_Control.md](11_Reward_Control.md) - Reward function
