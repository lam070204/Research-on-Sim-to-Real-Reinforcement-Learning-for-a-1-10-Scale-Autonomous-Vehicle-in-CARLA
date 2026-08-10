# 03 Installation

## Hướng dẫn Cài đặt

Hướng dẫn này hướng dẫn bạn cài đặt toàn bộ môi trường để chạy dự án Autonomous Driving trong CARLA.

## Yêu cầu Hệ thống

### Phần cứng
- **CPU**: Intel Core i7 hoặc tương đương (khuyến nghị i9/Ryzen 9)
- **RAM**: 16GB tối thiểu, 32GB khuyến nghị
- **GPU**: NVIDIA GPU với 8GB VRAM tối thiểu (RTX 3060 trở lên)
- **Dung lượng**: 50GB cho CARLA + 10GB cho dependencies + không gian cho dataset
- **Hệ điều hành**: Windows 10/11 (64-bit)

### Phần mềm
- **Python**: 3.8+
- **CARLA**: 0.9.13
- **CUDA**: 11.1+ (nếu dùng GPU NVIDIA)
- **cuDNN**: 8.0+

## Bước 1: Cài đặt Python

1. Tải Python 3.8+ từ https://www.python.org/downloads/
2. Trong quá trình cài đặt, tick chọn "Add Python to PATH"
3. Verify cài đặt:
```bash
python --version
pip --version
```

## Bước 2: Cài đặt CARLA 0.9.13

### Option A: Tải từ trang chủ CARLA
1. Truy cập https://carla.org/
2. Tải CARLA 0.9.13 cho Windows
3. Giải nén vào thư mục (ví dụ: `C:\CARLA_0.9.13`)

### Option B: Tải từ GitHub Releases
1. Truy cập https://github.com/carla-simulator/carla/releases
2. Tìm release 0.9.13
3. Tải `CARLA_0.9.13-win.zip`
4. Giải nén vào thư mục mong muốn

### Thêm CARLA Egg vào Python Path

CARLA Python API nằm trong file egg. Copy vào project:

```bash
# Copy CARLA egg vào thư mục carla/ của project
xcopy "C:\CARLA_0.9.13\PythonAPI\carla\dist\carla-0.9.13-py3.8-win-amd64.egg" "carla\" /Y
```

Hoặc thêm vào PYTHONPATH:
```bash
set PYTHONPATH=%PYTHONPATH%;C:\CARLA_0.9.13\PythonAPI\carla\dist\carla-0.9.13-py3.8-win-amd64.egg
```

### Test CARLA Installation

Chạy CARLA server:
```bash
cd C:\CARLA_0.9.13
CarlaUE4.exe
```

## Bước 3: Cài đặt Dependencies

### Option A: Dùng pip
```bash
pip install -r requirements.txt
```

### Option B: Dùng Poetry (khuyến nghị)
```bash
cd poetry
poetry install
poetry shell
```

### Cài đặt thủ công các package chính:
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install numpy==1.21.0
pip install pygame==2.1.0
pip install opencv-python==4.5.3
pip install tensorboard==2.10.0
pip install pillow==9.0.0
pip install tqdm==4.62.0
```

## Bước 4: Verify Ci đặt

### Test CARLA Connection
```bash
python -c "import sys; sys.path.append('carla'); import carla; print('CARLA version:', carla.__version__)"
```

### Test PyTorch với CUDA
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

### Test Import Project
```bash
python -c "from simulation.connection import carla; print('Connection OK')"
python -c "from autoencoder_rgb.encoder_rgb import VariationalEncoderRGB; print('VAE OK')"
python -c "from networks.on_policy.ppo.ppo import ActorCritic; print('PPO OK')"
```

## Bước 5: Cài đặt TensorBoard (Optional)

Để theo dõi quá trình training:
```bash
pip install tensorboard
```

Khởi chạy TensorBoard:
```bash
tensorboard --logdir=runs/
```

Truy cập http://localhost:6006 để xem logs.

## Xử lý Sự cố

### Lỗi: "No module named 'carla'"
```bash
# Thêm CARLA egg vào PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;carla\carla-0.9.13-py3.8-win-amd64.egg
```

### Lỗi: "CUDA not available"
- Kiểm tra CUDA installation: `nvcc --version`
- Cài đặt đúng phiên bản PyTorch với CUDA support
- Update NVIDIA driver

### Lỗi: "ImportError: DLL load failed"
- Cài đặt Microsoft Visual C++ Redistributable
- https://aka.ms/vs/16/release/vc_redist.x64.exe

### Lỗi: CARLA không khởi động được
- Kiểm tra GPU driver
- Chạy CARLA ở chế độ OpenGL: `CarlaUE4.exe -opengl`
- Giảm chất lượng graphics trong CARLA settings

## Kiểm tra Cài đặt Hoàn chỉnh

Chạy script test:
```bash
python -c "
import sys
sys.path.append('carla')
import carla
import torch
import numpy as np

print('=' * 50)
print('INSTALLATION CHECK')
print('=' * 50)
print(f'CARLA version: {carla.__version__}')
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'NumPy version: {np.__version__}')
print('=' * 50)
print('All checks passed!' if torch.cuda.is_available() else 'Warning: CUDA not available')
"
```

## Next Steps

Sau khi cài đặt xong:
1. Đọc [04_RoadRunner_Map.md](04_RoadRunner_Map.md) để tạo map tùy chỉnh
2. Đọc [06_Sensors.md](06_Sensors.md) để hiểu về hệ thống sensor
3. Đọc [08_VAE_RGB.md](08_VAE_RGB.md) để training VAE
4. Đọc [09_PPO.md](09_PPO.md) để training PPO agent
