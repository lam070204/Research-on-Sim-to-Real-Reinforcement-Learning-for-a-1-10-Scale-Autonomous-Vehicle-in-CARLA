# 15 Troubleshooting

## Hướng dẫn xử lý các vấn đề thường gặp

Tài liệu tổng hợp các lỗi thường gặp và cách khắc phục trong quá trình cài đặt, training và deployment.

## Mục lục

- [Cài đặt](#cài-đặt)
- [CARLA Simulation](#carla-simulation)
- [Training](#training)
- [VAE](#vae)
- [PPO](#ppo)
- [Memory Issues](#memory-issues)
- [Performance](#performance)

---

## Cài đặt

### Issue: Cannot install requirements

**Lỗi:**
```
ERROR: Could not find a version that satisfies the requirement torch==1.13.1
```

**Nguyên nhân:**
- Python version không tương thích
- pip version cũ

**Giải pháp:**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Check Python version (should be 3.7-3.9)
python --version

# Install with specific PyTorch version
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

### Issue: CARLA Python API import failed

**Lỗi:**
```
ModuleNotFoundError: No module named 'carla'
```

**Giải pháp:**
```bash
# Add CARLA egg file to Python path
cd carla
pip install carla-0.9.8-py3.7-win-amd64.egg

# Or manually add to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;c:\path\to\carla\carla-0.9.8-py3.7-win-amd64.egg
```

### Issue: CUDA not available

**Lỗi:**
```
CUDA unavailable: using CPU instead
```

**Giải pháp:**
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## CARLA Simulation

### Issue: Cannot connect to CARLA server

**Lỗi:**
```
Failed to connect to localhost:2000
```

**Giải pháp:**
1. Đảm bảo CARLA đang chạy
2. Kiểm tra port 2000 không bị chiếm dụng
3. Restart CARLA:
```bash
# Windows
CarlaUE4.exe -quality-level=Low

# Or with specific settings
CarlaUE4.exe -windowed -ResX=800 -ResY=600
```

### Issue: Vehicle spawning failed

**Lỗi:**
```
RuntimeError: unable to spawn actor
```

**Nguyên nhân:**
- Spawn point bị chặn
- Map chưa load xong

**Giải pháp:**
```python
# Wait for map to load
world.wait_for_tick(5.0)

# Try multiple spawn points
spawn_points = world.get_map().get_spawn_points()
for spawn_point in spawn_points:
    try:
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        break
    except:
        continue
```

### Issue: Sensors not working

**Lỗi:**
```
AttributeError: 'Camera' object has no attribute 'listen'
```

**Giải pháp:**
```python
# Ensure sensor is properly initialized
world.wait_for_tick(1.0)

# Check sensor blueprint
camera_bp = blueprint_library.find('sensor.camera.rgb')
if camera_bp is None:
    print("Camera blueprint not found!")
    
# Proper sensor setup
camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
camera.listen(callback)
world.wait_for_tick(0.5)
```

---

## Training

### Issue: Training diverges

**Triệu chứng:**
- Reward giảm mạnh
- Loss tăng vô hạn
- Agent không học được gì

**Nguyên nhân:**
- Learning rate quá cao
- Reward function không phù hợp
- Exploration quá ít

**Giải pháp:**
```python
# Reduce learning rate
learning_rate = 1e-4  # Instead of 3e-4

# Clip gradients
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)

# Adjust reward scaling
reward = np.clip(reward, -1, 1)

# Increase exploration
clip_epsilon = 0.3  # Instead of 0.2
```

### Issue: Slow training

**Triệu chứng:**
- FPS thấp (< 100)
- Training mất nhiều thời gian

**Giải pháp:**
```python
# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Reduce image resolution
image_size = (80, 160)  # Instead of higher

# Use smaller batch size
batch_size = 32  # Instead of 64

# Optimize data loading
num_workers = 4
pin_memory = True
```

### Issue: Episode always ends early

**Triệu chứng:**
- Episode length < 100 steps
- Frequent collisions/off-road

**Nguyên nhân:**
- Reward function quá khắc nghiệt
- Initial conditions khó

**Giải pháp:**
```python
# Relax termination conditions
LANE_LIMIT = 3.5  # Instead of 3.0
HEADING_LIMIT = 25  # Instead of 20

# Easier spawn points
spawn_points = [p for p in world.get_map().get_spawn_points() 
                if is_safe_spawn(p)]

# Curriculum learning
start with easy scenarios, gradually increase difficulty
```

---

## VAE

### Issue: VAE reconstruction quality poor

**Triệu chứng:**
- Ảnh reconstruction mờ
- Mất chi tiết quan trọng

**Nguyên nhân:**
- KL weight quá cao
- Model capacity thấp
- Training chưa đủ

**Giải pháp:**
```python
# Adjust KL weight
kl_weight = 0.1  # Instead of 1.0

# Increase model capacity
hidden_dim = 1024  # Instead of 512
latent_dim = 128  # Instead of 64

# Train longer
num_epochs = 200  # Instead of 100

# Use KL annealing
kl_weight = min(1.0, epoch / 50)
```

### Issue: VAE training loss NaN

**Lỗi:**
```
Loss: nan
```

**Nguyên nhân:**
- Learning rate quá cao
- Gradient explosion
- Data normalization sai

**Giải pháp:**
```python
# Reduce learning rate
learning_rate = 1e-4

# Gradient clipping
torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)

# Check data normalization
image = image.astype(np.float32) / 255.0  # [0, 1]
image = image.flatten()

# Add epsilon to log
kl_loss = -0.5 * torch.sum(1 + torch.log(sigma2 + 1e-8) - mu**2 - sigma2)
```

### Issue: Out of memory during VAE training

**Lỗi:**
```
RuntimeError: CUDA out of memory
```

**Giải pháp:**
```python
# Reduce batch size
batch_size = 32  # Instead of 128

# Gradient accumulation
accumulate_steps = 4
for i, batch in enumerate(loader):
    loss = compute_loss(batch)
    loss = loss / accumulate_steps
    loss.backward()
    
    if (i + 1) % accumulate_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Clear cache
torch.cuda.empty_cache()
```

---

## PPO

### Issue: PPO policy collapse

**Triệu chứng:**
- Reward đột ngột giảm về 0
- Policy không recover được

**Nguyên nhân:**
- KL divergence quá cao
- Update quá nhiều steps

**Giải pháp:**
```python
# Early stopping based on KL
kl_div = compute_kl(old_policy, new_policy)
if kl_div > 2 * target_kl:
    break  # Stop PPO updates

# Reduce update epochs
ppo_epochs = 5  # Instead of 10

# Increase clip epsilon
clip_epsilon = 0.3  # Instead of 0.2

# Add entropy bonus
entropy_bonus = 0.01
loss = policy_loss - entropy_bonus * entropy
```

### Issue: Action always the same

**Triệu chứng:**
- Agent luôn chọn cùng 1 action
- Không có exploration

**Nguyên nhân:**
- Action std quá nhỏ
- Policy collapse

**Giải pháp:**
```python
# Minimum action std
action_std = torch.clamp(action_std, min=0.1)

# Add exploration noise
action = action_mean + action_std * noise
noise = torch.randn_like(action_mean)

# Entropy regularization
entropy = normal.entropy().mean()
loss = policy_loss - 0.01 * entropy
```

### Issue: Value function overfitting

**Triệu chứng:**
- Value loss rất thấp nhưng performance kém
- Value predictions không chính xác

**Giải pháp:**
```python
# Value function clipping
value_pred_clipped = value_old + (value - value_old).clamp(-clip_epsilon, clip_epsilon)
value_loss = 0.5 * ((value_pred - returns)**2)
value_loss_clipped = 0.5 * ((value_pred_clipped - returns)**2)
value_loss = torch.max(value_loss, value_loss_clipped)

# Reduce value loss weight
c2 = 0.25  # Instead of 0.5

# Normalize returns
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

---

## Memory Issues

### Issue: CUDA Out of Memory

**Lỗi:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Giải pháp:**
```python
# Reduce batch size
batch_size = 16  # Instead of 64

# Clear cache regularly
torch.cuda.empty_cache()

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = compute_loss(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Monitor GPU usage
nvidia-smi
```

### Issue: Memory leak in training loop

**Triệu chứng:**
- GPU memory tăng dần theo thời gian
- Eventually OOM

**Giải pháp:**
```python
# Detach tensors
loss = loss.detach()

# Clear computation graph
optimizer.zero_grad()

# Use context manager
with torch.no_grad():
    for inference

# Delete unused variables
del buffer
buffer = []

# Profile memory
import torch.cuda as cuda
print(f"Allocated: {cuda.memory_allocated()/1e6:.1f}MB")
print(f"Cached: {cuda.memory_reserved()/1e6:.1f}MB")
```

---

## Performance

### Issue: Low FPS during inference

**Triệu chứng:**
- FPS < 20
- Control không mượt

**Giải pháp:**
```python
# Use ONNX runtime
import onnxruntime as ort
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# Optimize model
torch.jit.script(model)

# Reduce image resolution
image_size = (40, 80)  # Instead of (80, 160)

# Use smaller model
hidden_dim = 128  # Instead of 256

# Batch inference
actions = policy(batch_of_states)
```

### Issue: High latency

**Triệu chứng:**
- Delay giữa observation và action
- Control không responsive

**Giải pháp:**
```python
# Async sensor reading
sensor_data_queue = deque(maxlen=1)

# Parallel processing
from multiprocessing import Process

# Optimize preprocessing
image = np.frombuffer(image.raw_data, dtype=np.uint8)
image = image.reshape((80, 160, 4))[:, :, :3]
image = image.astype(np.float32) / 255.0

# Use TensorRT
from torch2trt import torch2trt
model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
```

---

## Debugging Tools

### Memory Profiler

```python
import torch
from torch.cuda import memory_summary

# Print memory summary
print(memory_summary())

# Profile memory allocation
torch.cuda.memory._record_memory_history(True)

# ... run training ...

# Get snapshot
snapshot = torch.cuda.memory._record_memory_history()
torch.cuda.memory._dump_snapshot("snapshot.pickle")

# Visualize at: https://pytorch.org/memory_viz
```

### Training Debugger

```python
class TrainingDebugger:
    def __init__(self):
        self.grad_norms = []
        self.losses = []
        self.rewards = []
    
    def log(self, loss, grads, reward, step):
        self.losses.append(loss)
        self.grad_norms.append(grads)
        self.rewards.append(reward)
        
        # Check for issues
        if np.isnan(loss):
            print(f"NaN loss at step {step}!")
        if grads > 10:
            print(f"Large gradients at step {step}: {grads}")
        if reward < -5:
            print(f"Very negative reward at step {step}: {reward}")
    
    def plot(self):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        axes[0].plot(self.losses)
        axes[0].set_title('Loss')
        
        axes[1].plot(self.grad_norms)
        axes[1].set_title('Gradient Norm')
        
        axes[2].plot(self.rewards)
        axes[2].set_title('Reward')
        
        plt.tight_layout()
        plt.savefig('debug_plot.png')
```

### CARLA Debugger

```python
class CARLADebugger:
    def __init__(self, world):
        self.world = world
        self.debug = world.debug
    
    def draw_waypoints(self, route):
        for i, waypoint in enumerate(route):
            location = waypoint.transform.location
            self.debug.draw_point(
                location,
                size=0.1,
                life_time=0.1,
                color=carla.Color(255, 0, 0)
            )
    
    def draw_vehicle_info(self, vehicle):
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        location = vehicle.get_location()
        location.z += 2.0
        
        self.debug.draw_string(
            location,
            f"Speed: {speed*3.6:.1f} km/h",
            size=0.5,
            life_time=0.1,
            color=carla.Color(0, 255, 0)
        )
```

---

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| CUDA OOM | Reduce batch size, clear cache |
| Training diverges | Lower learning rate, clip gradients |
| Poor reconstruction | Reduce KL weight, train longer |
| Policy collapse | Increase clip epsilon, add entropy |
| Slow training | Use GPU, reduce image size |
| Cannot connect to CARLA | Restart CARLA, check port 2000 |
| Import errors | Check Python version, reinstall requirements |

---

## Getting Help

### Resources

- [CARLA Documentation](https://carla.readthedocs.io/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [Stable Baselines3 Issues](https://github.com/DLR-RM/stable-baselines3/issues)

### Debug Checklist

- [ ] Check Python version (3.7-3.9)
- [ ] Verify CUDA installation
- [ ] Ensure CARLA is running
- [ ] Check GPU memory
- [ ] Review error messages
- [ ] Check file paths
- [ ] Verify model checkpoint format

---

## Next Steps

- [16_API.md](16_API.md) - API reference
- [17_Developer_Guide.md](17_Developer_Guide.md) - Developer guide
- [01_Project_Overview.md](01_Project_Overview.md) - Project overview
