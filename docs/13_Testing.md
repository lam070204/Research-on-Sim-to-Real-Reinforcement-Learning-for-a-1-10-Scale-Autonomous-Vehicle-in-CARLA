# 13 Testing

## Testing Agent Performance

Tài liệu về testing và đánh giá agent sau khi training.

## Tổng quan

Testing là bước quan trọng để đánh giá performance của agent trong các scenarios khác nhau. Testing giúp:
- Đánh giá khả năng generalization
- Phát hiện các edge cases
- So sánh các versions
- Validate trước khi deployment

## Testing Setup

### Test Environment

```python
import carla
import numpy as np
import time

class TestEnvironment:
    def __init__(self, town='Town07', weather='Clear'):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        
        # Set weather
        self.set_weather(weather)
        
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        
    def set_weather(self, weather):
        """Set weather conditions for testing."""
        weather_presets = {
            'Clear': carla.WeatherParameters.ClearNoon,
            'Cloudy': carla.WeatherParameters.CloudyNoon,
            'Rain': carla.WeatherParameters.MidRainyNoon,
            'Fog': carla.WeatherParameters.HardFogNoon,
            'Night': carla.WeatherParameters.ClearNight
        }
        self.world.set_weather(weather_presets.get(weather, carla.WeatherParameters.ClearNoon))
    
    def spawn_vehicle(self, spawn_point=None):
        """Spawn test vehicle."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.lincoln*')[0]
        
        if spawn_point is None:
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
        
        # Attach collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        
        self.collision_count = 0
        self.collision_sensor.listen(self._on_collision)
    
    def _on_collision(self, event):
        self.collision_count += 1
    
    def get_camera_image(self):
        """Get current camera image."""
        # Implementation depends on how you capture images
        pass
    
    def cleanup(self):
        """Cleanup actors."""
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
```

### Load Trained Agent

```python
import torch
from networks.on_policy.ppo.agent_v2 import PPOAgent

def load_test_agent(checkpoint_path, device='cuda'):
    """Load trained agent for testing."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    agent = PPOAgent(
        latent_dim=95,
        nav_dim=5,
        action_dim=2,
        hidden_dim=256,
        device=device
    )
    
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.policy.eval()  # Set to evaluation mode
    
    return agent
```

### Load VAE Encoder

```python
from autoencoder_rgb.vae_rgb import VAE
from encoder_init_rgb_v2 import EncodeStateRGBV2

def load_test_encoder(vae_path, device='cuda'):
    """Load trained VAE encoder."""
    encoder = EncodeStateRGBV2(latent_dim=95, device=device)
    encoder.load_weights(vae_path)
    encoder.eval()
    
    return encoder
```

## Testing Scenarios

### Scenario 1: Straight Road

```python
def test_straight_road(agent, encoder, env, distance=100):
    """
    Test agent on straight road.
    
    Metrics:
    - Time to complete
    - Lane keeping accuracy
    - Speed consistency
    """
    env.spawn_vehicle()
    
    start_time = time.time()
    distances = []
    speeds = []
    
    while True:
        # Get observation
        rgb_image = env.get_camera_image()
        latent = encoder.process_observation(rgb_image)
        
        # Get action (deterministic for testing)
        action, _ = agent.select_action((latent, nav_state), deterministic=True)
        
        # Apply action
        env.vehicle.apply_control(carla.VehicleControl(
            steer=float(action[0]),
            throttle=float(action[1])
        ))
        
        # Record metrics
        velocity = env.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speeds.append(speed)
        
        location = env.vehicle.get_location()
        distances.append(location.x)
        
        # Check completion
        if location.x > distance:
            success = True
            break
        
        # Check failure
        if env.collision_count > 0:
            success = False
            break
        
        time.sleep(0.05)
    
    env.cleanup()
    
    return {
        'success': success,
        'time': time.time() - start_time,
        'avg_speed': np.mean(speeds),
        'std_speed': np.std(speeds)
    }
```

### Scenario 2: Curved Road

```python
def test_curved_road(agent, encoder, env, curve_radius=50):
    """
    Test agent on curved road.
    
    Metrics:
    - Lane keeping on curves
    - Speed adjustment
    - Completion success
    """
    # Similar structure to straight road test
    # But evaluate on curved sections
    pass
```

### Scenario 3: Intersection

```python
def test_intersection(agent, encoder, env):
    """
    Test agent at intersection.
    
    Metrics:
    - Decision making
    - Lane selection
    - Collision avoidance
    """
    pass
```

### Scenario 4: Weather Conditions

```python
def test_weather_robustness(agent, encoder, weathers=['Clear', 'Rain', 'Fog', 'Night']):
    """
    Test agent under different weather conditions.
    
    Args:
        weathers: List of weather conditions to test
    """
    results = {}
    
    for weather in weathers:
        env = TestEnvironment(town='Town07', weather=weather)
        
        # Run multiple trials
        trials = []
        for i in range(10):
            result = test_straight_road(agent, encoder, env)
            trials.append(result)
            env.cleanup()
        
        results[weather] = {
            'success_rate': np.mean([t['success'] for t in trials]),
            'avg_time': np.mean([t['time'] for t in trials]),
            'avg_speed': np.mean([t['avg_speed'] for t in trials])
        }
    
    return results
```

## Testing Script

### Full Test Suite

```python
# test_agent.py

import argparse
import torch
import numpy as np
import json
from datetime import datetime

def run_test_suite(checkpoint_path, vae_path, num_trials=10):
    """
    Run comprehensive test suite.
    
    Args:
        checkpoint_path: Path to PPO checkpoint
        vae_path: Path to VAE weights
        num_trials: Number of trials per scenario
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load models
    print("Loading models...")
    agent = load_test_agent(checkpoint_path, device)
    encoder = load_test_encoder(vae_path, device)
    
    # Initialize test environment
    env = TestEnvironment(town='Town07')
    
    # Test scenarios
    scenarios = {
        'straight_clear': lambda: test_straight_road(agent, encoder, env),
        'straight_rain': lambda: test_straight_road(
            agent, encoder, 
            TestEnvironment(town='Town07', weather='Rain')
        ),
        'straight_night': lambda: test_straight_road(
            agent, encoder,
            TestEnvironment(town='Town07', weather='Night')
        ),
        'curved': lambda: test_curved_road(agent, encoder, env),
        'intersection': lambda: test_intersection(agent, encoder, env)
    }
    
    results = {}
    
    for name, scenario in scenarios.items():
        print(f"\nTesting {name}...")
        
        trials = []
        for i in range(num_trials):
            try:
                result = scenario()
                trials.append(result)
                print(f"  Trial {i+1}/{num_trials}: {'Success' if result['success'] else 'Failed'}")
            except Exception as e:
                print(f"  Trial {i+1}/{num_trials}: Error - {e}")
                trials.append({'success': False, 'error': str(e)})
        
        # Aggregate results
        successes = [t for t in trials if t.get('success', False)]
        
        results[name] = {
            'success_rate': len(successes) / num_trials,
            'avg_time': np.mean([t.get('time', 0) for t in trials]),
            'avg_speed': np.mean([t.get('avg_speed', 0) for t in trials]),
            'std_speed': np.mean([t.get('std_speed', 0) for t in trials]),
            'trials': num_trials,
            'errors': len([t for t in trials if 'error' in t])
        }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f'test_results_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"VAE: {vae_path}")
    print(f"{'='*60}")
    
    for scenario, metrics in results.items():
        print(f"\n{scenario.upper()}:")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"  Avg Time: {metrics['avg_time']:.2f}s")
        print(f"  Avg Speed: {metrics['avg_speed']:.2f} m/s")
        print(f"  Errors: {metrics['errors']}/{num_trials}")
    
    print(f"\nResults saved to: {results_path}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vae', type=str, default='autoencoder_rgb/model/vae_rgb_epoch_100.pth')
    parser.add_argument('--trials', type=int, default=10)
    
    args = parser.parse_args()
    
    run_test_suite(args.checkpoint, args.vae, args.trials)
```

## Evaluation Metrics

### Success Rate

```python
def calculate_success_rate(trials):
    """Calculate success rate from trials."""
    successes = sum(1 for t in trials if t['success'])
    return successes / len(trials)
```

### Average Completion Time

```python
def calculate_avg_time(trials):
    """Calculate average completion time."""
    successful_trials = [t for t in trials if t['success']]
    if not successful_trials:
        return float('inf')
    return np.mean([t['time'] for t in successful_trials])
```

### Lane Keeping Accuracy

```python
def calculate_lane_keeping_accuracy(trials):
    """Calculate lane keeping accuracy."""
    # Based on distance from center
    pass
```

### Collision Rate

```python
def calculate_collision_rate(trials):
    """Calculate collision rate."""
    collisions = sum(t.get('collision_count', 0) for t in trials)
    return collisions / len(trials)
```

## Visualization

### Plot Test Results

```python
import matplotlib.pyplot as plt

def plot_test_results(results):
    """Plot test results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenarios = list(results.keys())
    
    # Success Rate
    success_rates = [results[s]['success_rate'] for s in scenarios]
    axes[0, 0].bar(scenarios, success_rates)
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Average Time
    avg_times = [results[s]['avg_time'] for s in scenarios]
    axes[0, 1].bar(scenarios, avg_times)
    axes[0, 1].set_ylabel('Avg Time (s)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Average Speed
    avg_speeds = [results[s]['avg_speed'] for s in scenarios]
    axes[1, 0].bar(scenarios, avg_speeds)
    axes[1, 0].set_ylabel('Avg Speed (m/s)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Error Rate
    error_rates = [results[s]['errors'] / results[s]['trials'] for s in scenarios]
    axes[1, 1].bar(scenarios, error_rates)
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150)
    plt.show()
```

### Trajectory Visualization

```python
def plot_trajectory(locations):
    """Plot vehicle trajectory."""
    x = [loc.x for loc in locations]
    y = [loc.y for loc in locations]
    
    plt.figure(figsize=(10, 10))
    plt.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    plt.scatter(x[0], y[0], c='green', s=100, label='Start')
    plt.scatter(x[-1], y[-1], c='red', s=100, label='End')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Vehicle Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory.png')
    plt.show()
```

## Comparison Testing

### Compare Multiple Checkpoints

```python
def compare_checkpoints(checkpoint_paths, vae_path, num_trials=10):
    """
    Compare multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        vae_path: Path to VAE weights
        num_trials: Number of trials per checkpoint
    """
    results = {}
    
    for path in checkpoint_paths:
        print(f"\nTesting {path}...")
        result = run_test_suite(path, vae_path, num_trials)
        results[os.path.basename(path)] = result
    
    # Create comparison table
    import pandas as pd
    
    comparison_data = []
    for name, res in results.items():
        for scenario, metrics in res.items():
            comparison_data.append({
                'Checkpoint': name,
                'Scenario': scenario,
                'Success Rate': metrics['success_rate'],
                'Avg Time': metrics['avg_time'],
                'Avg Speed': metrics['avg_speed']
            })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    return df
```

## Best Practices

### 1. Deterministic Testing
```python
# Use deterministic actions during testing
action, _ = agent.select_action(state, deterministic=True)
```

### 2. Multiple Trials
```python
# Run multiple trials for statistical significance
num_trials = 30  # Minimum for statistical analysis
```

### 3. Diverse Scenarios
```python
# Test under various conditions
scenarios = ['clear', 'rain', 'night', 'fog', 'intersection', 'curve']
```

### 4. Record Everything
```python
# Save all test data for analysis
save_test_video = True
save_trajectories = True
save_metrics = True
```

### 5. Baseline Comparison
```python
# Compare with baseline (random/autopilot)
baseline_score = test_autopilot()
agent_score = test_agent()
improvement = (agent_score - baseline_score) / baseline_score
```

## Troubleshooting

### Issue: Agent performs poorly in testing
- Check if using deterministic mode
- Verify VAE reconstruction quality
- Review reward function alignment

### Issue: High variance in results
- Increase number of trials
- Check for random seed issues
- Ensure consistent initial conditions

### Issue: Memory issues during testing
- Clear CUDA cache between trials
- Use `torch.no_grad()` for inference
- Cleanup CARLA actors properly

## Next Steps

- [14_Deployment.md](14_Deployment.md) - Deployment
- [15_Troubleshooting.md](15_Troubleshooting.md) - Troubleshooting guide
- [12_Checkpoint.md](12_Checkpoint.md) - Checkpoint management
