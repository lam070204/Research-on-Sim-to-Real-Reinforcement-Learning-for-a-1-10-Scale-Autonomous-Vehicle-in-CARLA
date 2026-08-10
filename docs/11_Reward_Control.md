# 11 Reward Control

## Reward Function Design

Tài liệu về thiết kế và điều chỉnh reward function cho PPO training.

## Tổng quan

Reward function là một trong những thành phần quan trọng nhất trong Reinforcement Learning. Reward function tốt giúp agent học nhanh và đạt performance cao.

## Reward Function trong Dự án

### Công thức Reward

```python
def calculate_reward(self):
    vehicle_location = self.vehicle.get_location()
    velocity = self.vehicle.get_velocity()
    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # km/h
    
    # Get navigation errors
    distance_from_center, heading_error = self._calculate_navigation_errors()
    
    # Speed factor
    target_speed = 20  # km/h
    min_speed = 5
    max_speed = 30
    
    if speed < min_speed:
        speed_factor = speed / min_speed
    elif speed <= target_speed:
        speed_factor = 1.0
    else:
        speed_factor = max(1.0 - (speed - target_speed) / (max_speed - target_speed), 0.0)
    
    # Centering factor
    centering_factor = max(1.0 - abs(distance_from_center) / 3.0, 0.0)
    
    # Angle factor
    angle_factor = max(1.0 - abs(heading_error) / 20.0, 0.0)
    
    # Base reward
    reward = 1.2 * speed_factor * centering_factor * angle_factor
    
    # Heading bonus
    if abs(heading_error) < 5:
        reward += 0.4
    elif abs(heading_error) < 10:
        reward += 0.2
    
    # Lane deviation penalties
    if distance_from_center > 2.5:
        reward -= 2.0
    elif distance_from_center > 2.0:
        reward -= 1.0
    elif distance_from_center > 1.5:
        reward -= 0.5
    
    # Terminal rewards
    if self.collision_sensor.collision:
        reward = -10
    
    if distance_from_center > 3.0:
        reward = -10  # Off road
    
    if self.stuck_counter > 100:
        reward = -10  # Stuck
    
    if speed > 33:
        reward = -5  # Extreme overspeed
    
    # Route completion bonus
    if self.current_waypoint_index >= len(self.route) - 1:
        reward += 10.0
    
    return reward
```

## Reward Components

### 1. Speed Factor

```python
target_speed = 20  # km/h
min_speed = 5
max_speed = 30

if speed < min_speed:
    speed_factor = speed / min_speed  # Linear increase from 0 to 1
elif speed <= target_speed:
    speed_factor = 1.0  # Maximum reward at target speed
else:
    speed_factor = max(1.0 - (speed - target_speed) / (max_speed - target_speed), 0.0)
```

| Speed (km/h) | Speed Factor |
|--------------|--------------|
| 0 | 0.0 |
| 2.5 | 0.5 |
| 5 | 1.0 |
| 20 | 1.0 |
| 25 | 0.6 |
| 30 | 0.0 |
| >30 | 0.0 |

### 2. Centering Factor

```python
centering_factor = max(1.0 - abs(distance_from_center) / 3.0, 0.0)
```

| Distance from Center (m) | Centering Factor |
|--------------------------|------------------|
| 0.0 | 1.0 |
| 0.5 | 0.83 |
| 1.0 | 0.67 |
| 1.5 | 0.5 |
| 2.0 | 0.33 |
| 2.5 | 0.17 |
| 3.0 | 0.0 |

### 3. Angle Factor

```python
angle_factor = max(1.0 - abs(heading_error) / 20.0, 0.0)
```

| Heading Error (degrees) | Angle Factor |
|-------------------------|--------------|
| 0 | 1.0 |
| 5 | 0.75 |
| 10 | 0.5 |
| 15 | 0.25 |
| 20 | 0.0 |
| >20 | 0.0 |

### 4. Heading Bonus

```python
if abs(heading_error) < 5:
    reward += 0.4  # Excellent heading
elif abs(heading_error) < 10:
    reward += 0.2  # Good heading
```

### 5. Lane Deviation Penalties

```python
if distance_from_center > 2.5:
    reward -= 2.0  # Severe deviation
elif distance_from_center > 2.0:
    reward -= 1.0  # Moderate deviation
elif distance_from_center > 1.5:
    reward -= 0.5  # Mild deviation
```

### 6. Terminal Rewards

```python
# Collision penalty
if self.collision_sensor.collision:
    reward = -10

# Off-road penalty
if distance_from_center > 3.0:
    reward = -10

# Stuck penalty
if self.stuck_counter > 100:
    reward = -10

# Overspeed penalty
if speed > 33:
    reward = -5

# Route completion bonus
if self.current_waypoint_index >= len(self.route) - 1:
    reward += 10.0
```

## Reward Tuning Strategies

### Strategy 1: Dense Rewards

Dùng dense rewards để agent học nhanh hơn:

```python
# Dense reward - reward at every step
reward = (
    speed_factor * centering_factor * angle_factor +
    heading_bonus -
    lane_deviation_penalty
)
```

**Pros:**
- Agent học nhanh
- Ít bị sparse reward problem

**Cons:**
- Có thể overfit vào reward function
- Khó tune

### Strategy 2: Sparse Rewards

Chỉ reward khi đạt được mục tiêu:

```python
# Sparse reward - only at episode end
if done:
    if route_completed:
        reward = 100
    elif collision:
        reward = -10
    else:
        reward = 0
```

**Pros:**
- Đơn giản
- Tránh reward hacking

**Cons:**
- Học chậm
- Cần nhiều exploration

### Strategy 3: Shaped Rewards

Kết hợp dense và sparse rewards:

```python
# Shaped reward
reward = (
    0.01 * speed_factor +  # Small dense reward
    10.0 * route_progress  # Progress-based reward
    - 10.0 * collision     # Terminal penalty
)
```

## Hyperparameter Tuning

### Reward Weights

```python
# Reward weights
SPEED_WEIGHT = 1.2
CENTERING_WEIGHT = 1.0
ANGLE_WEIGHT = 1.0
HEADING_BONUS_EXCELLENT = 0.4
HEADING_BONUS_GOOD = 0.2
LANE_DEVIATION_SEVERE = -2.0
LANE_DEVIATION_MODERATE = -1.0
LANE_DEVIATION_MILD = -0.5
COLLISION_PENALTY = -10
OFF_ROAD_PENALTY = -10
STUCK_PENALTY = -10
OVERSPEED_PENALTY = -5
ROUTE_COMPLETION_BONUS = 10.0
```

### Speed Parameters

```python
# Speed parameters
TARGET_SPEED = 20  # km/h
MIN_SPEED = 5
MAX_SPEED = 30
OVERSPEED_THRESHOLD = 33
```

### Lane Parameters

```python
# Lane parameters
LANE_LIMIT = 3.0  # meters (off-road threshold)
LANE_DEVIATION_MILD = 1.5   # Start penalty
LANE_DEVIATION_MODERATE = 2.0
LANE_DEVIATION_SEVERE = 2.5
```

### Heading Parameters

```python
# Heading parameters
HEADING_LIMIT = 20.0  # degrees (zero factor threshold)
HEADING_EXCELLENT = 5   # Excellent heading bonus
HEADING_GOOD = 10       # Good heading bonus
```

## Reward Normalization

### Normalize Rewards

```python
class RewardNormalizer:
    def __init__(self, window_size=100):
        self.rewards = deque(maxlen=window_size)
    
    def normalize(self, reward):
        self.rewards.append(reward)
        
        if len(self.rewards) < 10:
            return reward
        
        mean = np.mean(self.rewards)
        std = np.std(self.rewards) + 1e-8
        
        return (reward - mean) / std
```

### Clip Rewards

```python
# Clip rewards to prevent extreme values
reward = np.clip(reward, -20, 20)
```

## Debugging Reward Function

### Log Reward Components

```python
def log_reward_components(self, reward_dict, step):
    writer.add_scalar('Reward/Speed', reward_dict['speed'], step)
    writer.add_scalar('Reward/Centering', reward_dict['centering'], step)
    writer.add_scalar('Reward/Angle', reward_dict['angle'], step)
    writer.add_scalar('Reward/Heading Bonus', reward_dict['heading_bonus'], step)
    writer.add_scalar('Reward/Lane Penalty', reward_dict['lane_penalty'], step)
    writer.add_scalar('Reward/Total', reward_dict['total'], step)
```

### Visualize Reward Distribution

```python
import matplotlib.pyplot as plt

def plot_reward_distribution(rewards):
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.savefig('reward_distribution.png')
    plt.show()
```

### Analyze Reward Trends

```python
def plot_reward_trend(rewards, window=100):
    rewards = np.array(rewards)
    
    # Moving average
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(ma, label=f'{window}-step MA')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Trend')
    plt.legend()
    plt.savefig('reward_trend.png')
    plt.show()
```

## Common Issues

### Issue 1: Reward Hacking

**Symptom:** Agent tìm cách maximize reward mà không thực sự học được task.

**Solution:**
- Review reward function
- Add constraints
- Use sparse rewards

### Issue 2: Sparse Rewards

**Symptom:** Agent không học được gì vì quá ít feedback.

**Solution:**
- Add shaping rewards
- Use curriculum learning
- Increase exploration

### Issue 3: Conflicting Rewards

**Symptom:** Agent không thể optimize tất cả rewards cùng lúc.

**Solution:**
- Prioritize rewards
- Adjust weights
- Simplify reward function

### Issue 4: Scale Issues

**Symptom:** Một số rewards quá lớn/nhỏ so với others.

**Solution:**
- Normalize rewards
- Adjust weights
- Clip extreme values

## Best Practices

### 1. Start Simple
Bắt đầu với reward function đơn giản, sau đó phức tạp dần.

### 2. Test Incrementally
Test từng component của reward function riêng biệt.

### 3. Monitor Training
Theo dõi reward trends và điều chỉnh khi cần.

### 4. Document Changes
Ghi lại tất cả changes và effects của chúng.

### 5. Validate Behavior
Đảm bảo agent học được behavior mong muốn, không chỉ maximize reward.

## Next Steps

- [09_PPO.md](09_PPO.md) - PPO training
- [12_Checkpoint.md](12_Checkpoint.md) - Checkpoint management
- [13_Testing.md](13_Testing.md) - Testing agent

</final_file_content>