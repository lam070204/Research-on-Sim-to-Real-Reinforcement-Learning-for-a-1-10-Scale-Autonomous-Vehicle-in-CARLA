# 14 Deployment

## Deploying Trained Models

Tài liệu về triển khai models đã trained cho autonomous driving trong simulation và real-world scenarios.

## Tổng quan

Deployment là quá trình đưa trained model vào hoạt động thực tế. Quy trình deployment bao gồm:
- Export models
- Setup inference environment
- Optimize cho real-time performance
- Deploy trên target hardware

## Export Models

### Export PPO Policy

```python
import torch
import onnx

def export_ppo_policy(checkpoint_path, output_path='deployed_models/ppo_policy.onnx'):
    """
    Export PPO policy to ONNX format.
    
    Args:
        checkpoint_path: Path to PPO checkpoint
        output_path: Path to save ONNX model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    from networks.on_policy.ppo.ppo import ActorCritic
    
    policy = ActorCritic(
        latent_dim=95,
        nav_dim=5,
        action_dim=2,
        hidden_dim=256
    )
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 100)  # latent (95) + nav (5)
    
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_mean', 'action_std', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action_mean': {0: 'batch_size'},
            'action_std': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    print(f"Exported PPO policy to {output_path}")

def export_ppo_to_torchscript(checkpoint_path, output_path='deployed_models/ppo_policy.pt'):
    """
    Export PPO policy to TorchScript format.
    
    Args:
        checkpoint_path: Path to PPO checkpoint
        output_path: Path to save TorchScript model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    from networks.on_policy.ppo.ppo import ActorCritic
    
    policy = ActorCritic(
        latent_dim=95,
        nav_dim=5,
        action_dim=2,
        hidden_dim=256
    )
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # Script the model
    scripted_policy = torch.jit.script(policy)
    scripted_policy.save(output_path)
    
    print(f"Exported PPO policy to {output_path}")
```

### Export VAE Encoder

```python
def export_vae_encoder(vae_path, output_path='deployed_models/vae_encoder.onnx'):
    """
    Export VAE encoder to ONNX format.
    
    Args:
        vae_path: Path to VAE checkpoint
        output_path: Path to save ONNX model
    """
    from autoencoder_rgb.vae_rgb import VAE
    
    vae = VAE(input_dim=38400, latent_dim=95, hidden_dim=512)
    
    checkpoint = torch.load(vae_path, map_location='cpu')
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    
    # Extract encoder
    encoder = vae.encoder
    
    # Export to ONNX
    dummy_input = torch.randn(1, 38400)  # Flattened 160x80x3 image
    
    torch.onnx.export(
        encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['image'],
        output_names=['latent'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'latent': {0: 'batch_size'}
        }
    )
    
    print(f"Exported VAE encoder to {output_path}")
```

### Export Complete Pipeline

```python
class AutonomousDrivingPipeline(torch.nn.Module):
    """Complete pipeline: Image -> Latent -> Action."""
    
    def __init__(self, encoder, policy):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
    
    def forward(self, image, nav_state):
        # Encode image
        latent = self.encoder(image)
        
        # Concatenate with navigation state
        full_state = torch.cat([latent, nav_state], dim=-1)
        
        # Get action
        action_mean, action_std, value = self.policy(full_state)
        
        return action_mean, action_std, value

def export_complete_pipeline(vae_path, ppo_path, output_path='deployed_models/full_pipeline.onnx'):
    """
    Export complete pipeline to ONNX.
    
    Args:
        vae_path: Path to VAE checkpoint
        ppo_path: Path to PPO checkpoint
        output_path: Path to save ONNX model
    """
    from autoencoder_rgb.vae_rgb import VAE
    from networks.on_policy.ppo.ppo import ActorCritic
    
    # Load models
    vae = VAE(input_dim=38400, latent_dim=95, hidden_dim=512)
    vae.load_state_dict(torch.load(vae_path, map_location='cpu')['model_state_dict'])
    vae.eval()
    
    policy = ActorCritic(latent_dim=95, nav_dim=5, action_dim=2, hidden_dim=256)
    policy.load_state_dict(torch.load(ppo_path, map_location='cpu')['policy_state_dict'])
    policy.eval()
    
    # Create pipeline
    pipeline = AutonomousDrivingPipeline(vae.encoder, policy)
    pipeline.eval()
    
    # Export
    dummy_image = torch.randn(1, 38400)
    dummy_nav = torch.randn(1, 5)
    
    torch.onnx.export(
        pipeline,
        (dummy_image, dummy_nav),
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['image', 'nav_state'],
        output_names=['action_mean', 'action_std', 'value']
    )
    
    print(f"Exported complete pipeline to {output_path}")
```

## Inference Engine

### Real-time Inference Class

```python
import torch
import numpy as np
import time
from collections import deque

class InferenceEngine:
    """Real-time inference engine for autonomous driving."""
    
    def __init__(
        self,
        policy_path,
        vae_path=None,
        device='cuda',
        use_onnx=False
    ):
        self.device = device
        
        if use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                policy_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        else:
            # Load PyTorch models
            self.policy = self._load_policy(policy_path)
            if vae_path:
                self.encoder = self._load_encoder(vae_path)
            else:
                self.encoder = None
        
        self.inference_times = deque(maxlen=100)
        self.fps = 0
    
    def _load_policy(self, path):
        """Load PPO policy."""
        if path.endswith('.pt'):
            return torch.jit.load(path, map_location=self.device)
        else:
            checkpoint = torch.load(path, map_location=self.device)
            policy = ActorCritic(latent_dim=95, nav_dim=5, action_dim=2)
            policy.load_state_dict(checkpoint['policy_state_dict'])
            policy.eval()
            return policy
    
    def _load_encoder(self, path):
        """Load VAE encoder."""
        encoder = EncodeStateRGBV2(latent_dim=95, device=self.device)
        encoder.load_weights(path)
        encoder.eval()
        return encoder
    
    def preprocess_image(self, image):
        """Preprocess image for inference."""
        # Resize if needed
        # Normalize
        # Flatten
        image = image.astype(np.float32) / 255.0
        image = image.flatten()
        return image
    
    def infer(self, image, nav_state, deterministic=True):
        """
        Run inference.
        
        Args:
            image: RGB image (80, 160, 3)
            nav_state: Navigation state (5,)
            deterministic: Use deterministic action
        
        Returns:
            action: (2,) numpy array
        """
        start_time = time.time()
        
        # Preprocess
        if self.encoder:
            latent = self.encoder.process_observation(image)
        else:
            latent = self.preprocess_image(image)
        
        full_state = np.concatenate([latent, nav_state])
        state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            if hasattr(self, 'session'):
                # ONNX inference
                outputs = self.session.run(
                    None,
                    {'state': state_tensor.cpu().numpy()}
                )
                action_mean = outputs[0]
            else:
                # PyTorch inference
                action_mean, action_std, value = self.policy.get_action(
                    state_tensor,
                    deterministic=deterministic
                )
                action_mean = action_mean.cpu().numpy()
        
        # Record inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        self.fps = 1000 / np.mean(self.inference_times)
        
        return action_mean.squeeze(0)
    
    def get_performance(self):
        """Get inference performance metrics."""
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'std_inference_time_ms': np.std(self.inference_times),
            'fps': self.fps
        }
```

### Deployment Script

```python
# deploy.py

import argparse
import carla
import numpy as np
import time
from inference_engine import InferenceEngine

class DeployedAgent:
    """Deployed agent for CARLA simulation."""
    
    def __init__(self, args):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(args.town)
        
        # Initialize inference engine
        self.engine = InferenceEngine(
            policy_path=args.policy,
            vae_path=args.vae,
            device=args.device,
            use_onnx=args.onnx
        )
        
        self.vehicle = None
        self.camera = None
        
    def setup_vehicle(self):
        """Setup vehicle and sensors."""
        blueprint_library = self.world.get_blueprint_library()
        
        # Spawn vehicle
        vehicle_bp = blueprint_library.filter('vehicle.lincoln*')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = np.random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Attach camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '160')
        camera_bp.set_attribute('image_size_y', '80')
        
        camera_transform = carla.Transform(
            carla.Location(x=2.5, y=0.0, z=1.5)
        )
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        self.current_image = None
        self.camera.listen(lambda image: self._on_image(image))
    
    def _on_image(self, image):
        """Callback for camera image."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.current_image = array[:, :, :3]  # RGB
    
    def get_nav_state(self):
        """Get navigation state."""
        # Implement navigation state extraction
        # Based on waypoint, speed, heading, etc.
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return np.array([
            speed,  # Current speed
            0.0,    # Distance to next waypoint
            0.0,    # Angle to next waypoint
            0.0,    # Lane offset
            0.0     # Heading error
        ])
    
    def run(self, duration=60):
        """Run deployed agent."""
        print("Starting deployed agent...")
        print(f"Inference FPS: {self.engine.fps:.1f}")
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            if self.current_image is None:
                time.sleep(0.1)
                continue
            
            # Get observation
            nav_state = self.get_nav_state()
            
            # Get action
            action = self.engine.infer(
                self.current_image,
                nav_state,
                deterministic=True
            )
            
            # Apply action
            steer, throttle = action
            self.vehicle.apply_control(
                carla.VehicleControl(
                    steer=float(np.clip(steer, -1, 1)),
                    throttle=float(np.clip(throttle, 0, 1))
                )
            )
            
            # Log performance
            if int(time.time() - start_time) % 10 == 0:
                perf = self.engine.get_performance()
                print(f"FPS: {perf['fps']:.1f}, Latency: {perf['avg_inference_time_ms']:.2f}ms")
        
        print("Deployment complete")
        self.cleanup()
    
    def cleanup(self):
        """Cleanup actors."""
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--vae', type=str, default=None)
    parser.add_argument('--town', type=str, default='Town07')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--duration', type=int, default=60)
    
    args = parser.parse_args()
    
    agent = DeployedAgent(args)
    agent.setup_vehicle()
    agent.run(duration=args.duration)
```

## Optimization

### TensorRT Optimization

```python
import torch
from torch2trt import torch2trt

def optimize_with_tensorrt(model_path, output_path='deployed_models/policy_trt.pth'):
    """
    Optimize model with TensorRT.
    
    Args:
        model_path: Path to PyTorch model
        output_path: Path to save optimized model
    """
    # Load model
    policy = ActorCritic(latent_dim=95, nav_dim=5, action_dim=2)
    policy.load_state_dict(torch.load(model_path, map_location='cpu')['policy_state_dict'])
    policy.eval()
    policy.cuda()
    
    # Create dummy input
    dummy_input = torch.randn(1, 100).cuda()
    
    # Convert to TensorRT
    model_trt = torch2trt(
        policy,
        [dummy_input],
        fp16_mode=True,
        max_batch_size=1,
        max_workspace_size=1 << 26
    )
    
    # Save optimized model
    torch.save(model_trt.state_dict(), output_path)
    print(f"Saved TensorRT optimized model to {output_path}")
```

### Quantization

```python
import torch.quantization as quantization

def quantize_model(model_path, output_path='deployed_models/policy_quantized.pt'):
    """
    Quantize model to INT8.
    
    Args:
        model_path: Path to PyTorch model
        output_path: Path to save quantized model
    """
    # Load model
    policy = ActorCritic(latent_dim=95, nav_dim=5, action_dim=2)
    policy.load_state_dict(torch.load(model_path, map_location='cpu')['policy_state_dict'])
    policy.eval()
    
    # Quantize
    policy_quantized = quantization.quantize_dynamic(
        policy,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    
    # Save
    torch.save(policy_quantized.state_dict(), output_path)
    print(f"Saved quantized model to {output_path}")
```

### Benchmark

```python
def benchmark_model(model_path, device='cuda', num_runs=1000):
    """
    Benchmark model inference time.
    
    Args:
        model_path: Path to model
        device: Device to run on
        num_runs: Number of runs
    """
    # Load model
    if model_path.endswith('.onnx'):
        import onnxruntime as ort
        session = ort.InferenceSession(model_path)
        
        def run(x):
            return session.run(None, {'state': x.cpu().numpy()})
    else:
        policy = ActorCritic(latent_dim=95, nav_dim=5, action_dim=2)
        policy.load_state_dict(torch.load(model_path, map_location=device)['policy_state_dict'])
        policy.eval()
        policy.to(device)
        
        def run(x):
            with torch.no_grad():
                return policy(x)
    
    # Warm up
    dummy_input = torch.randn(1, 100).to(device)
    for _ in range(100):
        run(dummy_input)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_runs):
        run(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / num_runs * 1000  # ms
    
    print(f"Benchmark Results:")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print(f"  Runs: {num_runs}")
    print(f"  Avg Inference Time: {avg_time:.3f}ms")
    print(f"  FPS: {1000/avg_time:.1f}")
    
    return avg_time
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy models
COPY deployed_models/ /app/models/

# Copy inference code
COPY inference_engine.py /app/
COPY deploy.py /app/

WORKDIR /app

# Run inference
CMD ["python3", "deploy.py", "--policy", "/app/models/ppo_policy.pt"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  autonomous-agent:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./deployed_models:/app/models
    network_mode: "host"
    command: >
      python3 deploy.py
      --policy /app/models/ppo_policy.pt
      --vae /app/models/vae_encoder.pt
      --town Town07
      --device cuda
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Inference Time | < 10ms | Real-time requirement |
| FPS | > 30 | Smooth control |
| Model Size | < 50MB | Easy deployment |
| Memory Usage | < 500MB | Edge device compatible |
| CPU Usage | < 50% | Leave room for other processes |

## Troubleshooting

### Issue: Slow inference
- Use ONNX runtime
- Enable TensorRT optimization
- Reduce model size
- Use quantization

### Issue: Model loading fails
- Check PyTorch version compatibility
- Verify checkpoint format
- Check device (CPU/GPU)

### Issue: High memory usage
- Use model quantization
- Reduce batch size
- Clear CUDA cache regularly

## Next Steps

- [13_Testing.md](13_Testing.md) - Testing before deployment
- [15_Troubleshooting.md](15_Troubleshooting.md) - Troubleshooting guide
- [16_API.md](16_API.md) - API reference
