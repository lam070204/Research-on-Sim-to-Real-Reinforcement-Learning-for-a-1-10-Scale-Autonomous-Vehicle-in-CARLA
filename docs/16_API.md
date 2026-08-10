# 16 API Reference

## API Documentation

Tài liệu tham khảo API cho các modules chính trong dự án.

## Mục lục

- [Environment API](#environment-api)
- [PPO Agent API](#ppo-agent-api)
- [VAE API](#vae-api)
- [Sensors API](#sensors-api)
- [Utilities API](#utilities-api)

---

## Environment API

### CarlaEnvironmentRGB

```python
from simulation.environment_rgb import CarlaEnvironmentRGB

env = CarlaEnvironmentRGB(
    town='Town07',
    port=2000,
    timeout=10.0
)
```

#### Methods

##### `reset()`
Reset environment và trả về initial state.

```python
state = env.reset()
# Returns: (latent, nav_state)
# - latent: np.array (95,)
# - nav_state: np.array (5,) [speed, distance, angle, lane_offset, heading_error]
```

##### `step(action)`
Thực hiện action và trả về kết quả.

```python
next_state, reward, done, info = env.step(action)
# Args:
#   action: np.array (2,) [steer, throttle]
# Returns:
#   next_state: (latent, nav_state)
#   reward: float
#   done: bool
#   info: dict {
#       'collision': bool,
#       'off_road': bool,
#       'stuck': bool,
#       'route_progress': float,
#       'speed': float
#   }
```

##### `render()`
Render current frame (optional).

```python
env.render(mode='human')
```

##### `close()`
Cleanup environment.

```python
env.close()
```

##### `get_route()`
Trả về route hiện tại.

```python
route = env.get_route()
# Returns: List of carla.Location
```

##### `get_spawn_points()`
Trả về các spawn points có sẵn.

```python
spawn_points = env.get_spawn_points()
# Returns: List of carla.Transform
```

---

## PPO Agent API

### PPOAgent

```python
from networks.on_policy.ppo.agent_v2 import PPOAgent

agent = PPOAgent(
    latent_dim=95,
    nav_dim=5,
    action_dim=2,
    hidden_dim=256,
    device='cuda'
)
```

#### Methods

##### `select_action(state, deterministic=False)`
Chọn action dựa trên policy.

```python
action, log_prob = agent.select_action(state, deterministic=False)
# Args:
#   state: tuple (latent, nav_state)
#   deterministic: bool (use mean action if True)
# Returns:
#   action: np.array (2,)
#   log_prob: float
```

##### `evaluate_actions(states, actions)`
Evaluate log probs và values cho batch actions.

```python
log_probs, values = agent.evaluate_actions(states, actions)
# Args:
#   states: torch.Tensor (batch, state_dim)
#   actions: torch.Tensor (batch, action_dim)
# Returns:
#   log_probs: torch.Tensor (batch,)
#   values: torch.Tensor (batch,)
```

##### `update(buffer)`
Update policy using PPO.

```python
losses = agent.update(buffer)
# Args:
#   buffer: List of tuples (state, action, reward, next_state, done, log_prob)
# Returns:
#   losses: dict {
#       'policy_loss': float,
#       'value_loss': float,
#       'entropy': float,
#       'clip_fraction': float
#   }
```

##### `save(path)`
Save checkpoint.

```python
agent.save('checkpoints/PPO/ppo_episode_10000.pth')
```

##### `load(path)`
Load checkpoint.

```python
agent.load('checkpoints/PPO/ppo_episode_10000.pth')
```

### ActorCritic

```python
from networks.on_policy.ppo.ppo import ActorCritic

policy = ActorCritic(
    latent_dim=95,
    nav_dim=5,
    action_dim=2,
    hidden_dim=256
)
```

#### Methods

##### `forward(state)`
Forward pass through policy and value network.

```python
action_mean, action_logstd, value = policy(state)
# Args:
#   state: torch.Tensor (batch, state_dim)
# Returns:
#   action_mean: torch.Tensor (batch, action_dim)
#   action_logstd: torch.Tensor (batch, action_dim)
#   value: torch.Tensor (batch,)
```

##### `get_action(state, deterministic=False)`
Get action from policy.

```python
action, log_prob = policy.get_action(state, deterministic=False)
```

##### `get_value(state)`
Get value estimate.

```python
value = policy.get_value(state)
```

---

## VAE API

### VAE

```python
from autoencoder_rgb.vae_rgb import VAE

vae = VAE(
    input_dim=38400,  # 160 * 80 * 3
    latent_dim=95,
    hidden_dim=512
)
```

#### Methods

##### `encode(x)`
Encode input to latent distribution.

```python
mu, logvar = vae.encode(x)
# Args:
#   x: torch.Tensor (batch, input_dim)
# Returns:
#   mu: torch.Tensor (batch, latent_dim)
#   logvar: torch.Tensor (batch, latent_dim)
```

##### `reparameterize(mu, logvar)`
Reparameterization trick.

```python
z = vae.reparameterize(mu, logvar)
# Args:
#   mu: torch.Tensor (batch, latent_dim)
#   logvar: torch.Tensor (batch, latent_dim)
# Returns:
#   z: torch.Tensor (batch, latent_dim)
```

##### `decode(z)`
Decode latent to reconstruction.

```python
reconstruction = vae.decode(z)
# Args:
#   z: torch.Tensor (batch, latent_dim)
# Returns:
#   reconstruction: torch.Tensor (batch, input_dim)
```

##### `forward(x)`
Full VAE forward pass.

```python
reconstruction, mu, logvar = vae(x)
```

##### `loss_function(reconstruction, x, mu, logvar)`
Compute VAE loss.

```python
loss = vae.loss_function(reconstruction, x, mu, logvar)
# Returns: dict {
#   'total_loss': float,
#   'reconstruction_loss': float,
#   'kl_loss': float
# }
```

##### `save(path)`
Save checkpoint.

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, path)
```

### EncodeStateRGBV2

```python
from encoder_init_rgb_v2 import EncodeStateRGBV2

encoder = EncodeStateRGBV2(
    latent_dim=95,
    device='cuda'
)
encoder.load_weights('autoencoder_rgb/model/vae_rgb_epoch_100.pth')
```

#### Methods

##### `process_observation(rgb_image)`
Process RGB image to latent state.

```python
latent = encoder.process_observation(rgb_image)
# Args:
#   rgb_image: np.array (80, 160, 3)
# Returns:
#   latent: np.array (95,)
```

##### `process_batch(rgb_images)`
Process batch of images.

```python
latents = encoder.process_batch(rgb_images)
# Args:
#   rgb_images: np.array (batch, 80, 160, 3)
# Returns:
#   latents: np.array (batch, 95)
```

---

## Sensors API

### CameraSensor

```python
from simulation.sensors import CameraSensor

camera = CameraSensor(
    world=world,
    vehicle=vehicle,
    width=160,
    height=80,
    x=2.5,
    y=0.0,
    z=1.5
)
```

#### Methods

##### `listen(callback)`
Register callback for image data.

```python
camera.listen(lambda image: process_image(image))
```

##### `stop()`
Stop listening.

```python
camera.stop()
```

##### `get_latest_image()`
Get latest captured image.

```python
image = camera.get_latest_image()
# Returns: np.array (80, 160, 3) RGB
```

##### `destroy()`
Destroy sensor actor.

```python
camera.destroy()
```

### CollisionSensor

```python
from simulation.sensors import CollisionSensor

collision = CollisionSensor(
    world=world,
    vehicle=vehicle
)
```

#### Methods

##### `listen(callback)`
Register callback for collision events.

```python
collision.listen(lambda event: handle_collision(event))
```

##### `get_collision_count()`
Get total collision count.

```python
count = collision.get_collision_count()
# Returns: int
```

##### `reset()`
Reset collision count.

```python
collision.reset()
```

##### `destroy()`
Destroy sensor actor.

```python
collision.destroy()
```

### LaneInvasionSensor

```python
from simulation.sensors import LaneInvasionSensor

lane_sensor = LaneInvasionSensor(
    world=world,
    vehicle=vehicle
)
```

#### Methods

##### `listen(callback)`
Register callback for lane invasion events.

```python
lane_sensor.listen(lambda event: handle_lane_invasion(event))
```

##### `get_invasion_count()`
Get lane invasion count.

```python
count = lane_sensor.get_invasion_count()
```

##### `destroy()`
Destroy sensor actor.

```python
lane_sensor.destroy()
```

---

## Utilities API

### Navigation Utils

```python
from simulation.utils import NavigationUtils
```

##### `calculate_route_distance(route)`
Calculate total route distance.

```python
distance = NavigationUtils.calculate_route_distance(route)
# Returns: float (meters)
```

##### `calculate_heading_error(vehicle, route)`
Calculate heading error to next waypoint.

```python
heading_error = NavigationUtils.calculate_heading_error(vehicle, route)
# Returns: float (degrees)
```

##### `calculate_distance_from_center(vehicle, route)`
Calculate distance from lane center.

```python
distance = NavigationUtils.calculate_distance_from_center(vehicle, route)
# Returns: float (meters)
```

##### `get_route_progress(vehicle, route)`
Calculate route completion percentage.

```python
progress = NavigationUtils.get_route_progress(vehicle, route)
# Returns: float (0.0 to 1.0)
```

### Reward Utils

```python
from simulation.utils import RewardUtils
```

##### `calculate_speed_reward(speed, target_speed=20, min_speed=5, max_speed=30)`
Calculate speed component of reward.

```python
speed_factor = RewardUtils.calculate_speed_reward(speed)
# Returns: float (0.0 to 1.0)
```

##### `calculate_centering_reward(distance_from_center, lane_limit=3.0)`
Calculate centering component of reward.

```python
centering_factor = RewardUtils.calculate_centering_reward(distance_from_center)
# Returns: float (0.0 to 1.0)
```

##### `calculate_angle_reward(heading_error, angle_limit=20.0)`
Calculate heading angle component of reward.

```python
angle_factor = RewardUtils.calculate_angle_reward(heading_error)
# Returns: float (0.0 to 1.0)
```

##### `calculate_terminal_reward(collision, off_road, stuck, completed)`
Calculate terminal reward.

```python
terminal_reward = RewardUtils.calculate_terminal_reward(
    collision=False,
    off_road=False,
    stuck=False,
    completed=True
)
# Returns: float
```

### Training Utils

```python
from networks.utils import TrainingUtils
```

##### `compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95)`
Compute Generalized Advantage Estimation.

```python
advantages, returns = TrainingUtils.compute_gae(
    rewards, values, next_value, gamma=0.99, lam=0.95
)
# Returns: (advantages, returns) - both np.arrays
```

##### `normalize_advantages(advantages)`
Normalize advantages to zero mean and unit variance.

```python
normalized = TrainingUtils.normalize_advantages(advantages)
```

##### `clip_gradients(model, max_norm=0.5)`
Clip gradients to prevent explosion.

```python
TrainingUtils.clip_gradients(model, max_norm=0.5)
```

##### `count_parameters(model)`
Count trainable parameters.

```python
num_params = TrainingUtils.count_parameters(model)
# Returns: int
```

### Visualization Utils

```python
from utils.visualization import VisualizationUtils
```

##### `plot_training_curve(rewards, window=100)`
Plot training reward curve.

```python
VisualizationUtils.plot_training_curve(rewards, window=100)
```

##### `plot_reconstruction(original, reconstruction)`
Plot VAE reconstruction comparison.

```python
VisualizationUtils.plot_reconstruction(original, reconstruction)
```

##### `plot_trajectory(locations)`
Plot vehicle trajectory.

```python
VisualizationUtils.plot_trajectory(locations)
```

##### `save_tensorboard_logs(writer, losses, rewards, step)`
Save logs to TensorBoard.

```python
VisualizationUtils.save_tensorboard_logs(writer, losses, rewards, step)
```

---

## Data Structures

### ReplayBuffer

```python
from networks.utils import ReplayBuffer

buffer = ReplayBuffer(capacity=10000)
```

##### `push(state, action, reward, next_state, done)`
Add transition to buffer.

```python
buffer.push(state, action, reward, next_state, done)
```

##### `sample(batch_size)`
Sample random batch.

```python
batch = buffer.sample(batch_size)
# Returns: dict with keys: states, actions, rewards, next_states, dones
```

##### `__len__()`
Get buffer size.

```python
size = len(buffer)
```

### RolloutBuffer

```python
from networks.on_policy.ppo.utils import RolloutBuffer

buffer = RolloutBuffer()
```

##### `add(state, action, reward, value, log_prob)`
Add transition to buffer.

```python
buffer.add(state, action, reward, value, log_prob)
```

##### `compute_returns_and_advantages(last_value, gamma=0.99, lam=0.95)`
Compute returns and advantages using GAE.

```python
returns, advantages = buffer.compute_returns_and_advantages(last_value)
```

##### `clear()`
Clear buffer.

```python
buffer.clear()
```

---

## Configuration

### Default Hyperparameters

```python
# PPO
PPO_CONFIG = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'c1': 0.5,  # Value loss weight
    'c2': 0.01,  # Entropy weight
    'ppo_epochs': 10,
    'batch_size': 64,
    'max_grad_norm': 0.5
}

# VAE
VAE_CONFIG = {
    'input_dim': 38400,
    'latent_dim': 95,
    'hidden_dim': 512,
    'learning_rate': 1e-4,
    'kl_weight': 1.0,
    'batch_size': 128,
    'num_epochs': 100
}

# Environment
ENV_CONFIG = {
    'town': 'Town07',
    'port': 2000,
    'image_width': 160,
    'image_height': 80,
    'camera_x': 2.5,
    'camera_y': 0.0,
    'camera_z': 1.5,
    'target_speed': 20,  # km/h
    'lane_limit': 3.0,  # meters
    'heading_limit': 20.0  # degrees
}
```

---

## Next Steps

- [17_Developer_Guide.md](17_Developer_Guide.md) - Developer guide
- [01_Project_Overview.md](01_Project_Overview.md) - Project overview
- [15_Troubleshooting.md](15_Troubleshooting.md) - Troubleshooting
