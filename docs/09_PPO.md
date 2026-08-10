# 09 PPO

## Proximal Policy Optimization

Tài liệu về PPO algorithm được sử dụng để training agent cho autonomous driving.

## Tổng quan

PPO (Proximal Policy Optimization) là một thuật toán Reinforcement Learning thuộc nhóm policy gradient methods, được phát triển bởi OpenAI. PPO đạt được sự cân bằng tốt giữa:
- **Sample efficiency**: Học được nhiều từ ít dữ liệu
- **Stability**: Tránh policy updates quá lớn
- **Performance**: Đạt kết quả tốt trên nhiều tasks

## Kiến trúc PPO trong Dự án

### Actor-Critic Network

File: `networks/on_policy/ppo/ppo.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, latent_dim=95, nav_dim=5, action_dim=2, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Input: latent (95) + navigation (5) = 100 dimensions
        self.input_dim = latent_dim + nav_dim
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Actor log variance (learnable parameter)
        self.actor_logvar = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: (batch, input_dim)
        h = self.backbone(x)
        
        # Policy
        action_mean = self.actor_mean(h)
        action_logvar = self.actor_logvar.expand_as(action_mean)
        action_std = torch.exp(0.5 * action_logvar)
        
        # Value
        value = self.critic(h)
        
        return action_mean, action_std, value
    
    def get_action(self, x, deterministic=False):
        action_mean, action_std, value = self.forward(x)
        
        if deterministic:
            return action_mean, value
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            return action, log_prob, value
    
    def evaluate_actions(self, x, actions):
        action_mean, action_std, value = self.forward(x)
        
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Entropy for regularization
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy
```

### PPO Agent

File: `networks/on_policy/ppo/agent_v2.py`

```python
import torch
import numpy as np
from collections import deque

class PPOAgent:
    def __init__(
        self,
        latent_dim=95,
        nav_dim=5,
        action_dim=2,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=0.5,  # Value loss coefficient
        c2=0.01,  # Entropy coefficient
        max_grad_norm=0.5,
        device='cuda'
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Actor-Critic network
        self.policy = ActorCritic(
            latent_dim=latent_dim,
            nav_dim=nav_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Training history
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def select_action(self, state, deterministic=False):
        """
        Select action given state.
        
        Args:
            state: tuple (rgb_image, navigation_state)
            deterministic: bool, use deterministic action during testing
        
        Returns:
            action: numpy array
        """
        rgb_image, nav_state = state
        
        # Process RGB image through encoder (already done in environment)
        # Concatenate latent + navigation
        full_state = np.concatenate([rgb_image, nav_state])
        state_tensor = torch.FloatTensor(full_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
        
        return action.cpu().numpy().squeeze(0), log_prob, value
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation.
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        return advantages, returns
    
    def update(self, buffer):
        """
        Update policy using PPO algorithm.
        
        Args:
            buffer: list of (state, action, log_prob, reward, done)
        """
        # Extract batch data
        states = torch.FloatTensor(np.array([b[0] for b in buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in buffer])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([b[2] for b in buffer])).to(self.device)
        rewards = [b[3] for b in buffer]
        dones = [b[4] for b in buffer]
        
        # Compute values and advantages
        with torch.no_grad():
            _, _, values = self.policy.get_action(states)
            values = values.squeeze(-1).cpu().numpy()
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update (multiple epochs)
        for _ in range(10):  # PPO epochs
            # Evaluate actions
            log_probs, values, entropies = self.policy.evaluate_actions(states, actions)
            
            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), returns)
            
            # Entropy bonus
            entropy_loss = -entropies.mean()
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'clip_fraction': (torch.abs(ratio - 1) > self.clip_epsilon).float().mean().item()
        }
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths)
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=100)
```

## Training Loop

File: `continuous_driver_rgb_v2.py`

```python
def train_ppo(
    num_episodes=1000000,
    steps_per_episode=1000,
    update_frequency=2048,  # Steps per update
    log_frequency=100,
    save_frequency=10000,
    checkpoint_path='checkpoints/PPO/'
):
    # Initialize
    env = CarlaEnvironmentRGB(town='Town07')
    encoder = EncodeStateRGBV2(latent_dim=95, device='cuda')
    encoder.load_weights('autoencoder_rgb/model/vae_rgb_epoch_100.pth')
    
    agent = PPOAgent(
        latent_dim=95,
        nav_dim=5,
        action_dim=2,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        device='cuda'
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir='runs/PPO_0.2_1000')
    
    buffer = []
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(steps_per_episode):
            # Get latent representation
            rgb_image, nav_state = state
            latent = encoder.process_observation(rgb_image)
            full_state = (latent, nav_state)
            
            # Select action
            action, log_prob, _ = agent.select_action(full_state)
            
            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            # Store in buffer
            buffer.append((
                np.concatenate([latent, nav_state]),
                action,
                log_prob.cpu().item(),
                reward,
                done
            ))
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            state = next_state
            
            # Update policy
            if len(buffer) >= update_frequency:
                metrics = agent.update(buffer)
                
                # Log metrics
                writer.add_scalar('Loss/Policy', metrics['policy_loss'], total_steps)
                writer.add_scalar('Loss/Value', metrics['value_loss'], total_steps)
                writer.add_scalar('Loss/Entropy', metrics['entropy'], total_steps)
                writer.add_scalar('Stats/Clip Fraction', metrics['clip_fraction'], total_steps)
                
                buffer.clear()
            
            if done:
                break
        
        # Track episode metrics
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        
        # Log episode metrics
        if episode % log_frequency == 0:
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/Length', episode_length, episode)
            writer.add_scalar('Episode/Avg Reward', np.mean(agent.episode_rewards), episode)
            writer.add_scalar('Episode/Avg Length', np.mean(agent.episode_lengths), episode)
            
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Length={episode_length}")
        
        # Save checkpoint
        if episode % save_frequency == 0:
            agent.save(f'{checkpoint_path}/ppo_episode_{episode}.pth')
    
    writer.close()
```

## Hyperparameters

| Hyperparameter | Giá trị | Mô tả |
|----------------|---------|-------|
| `latent_dim` | 95 | Chiều latent vector từ VAE |
| `nav_dim` | 5 | Chiều navigation state |
| `action_dim` | 2 | Steering + Throttle |
| `hidden_dim` | 256 | Chiều hidden layers |
| `learning_rate` | 3e-4 | Learning rate cho Adam |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda parameter |
| `clip_epsilon` | 0.2 | PPO clip parameter |
| `c1` | 0.5 | Value loss coefficient |
| `c2` | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping norm |
| `update_frequency` | 2048 | Steps per policy update |
| `ppo_epochs` | 10 | Số epochs per update |

## Action Space

### Continuous Action Space

```python
action = [steer, throttle]

# Steering: [-1, 1]
# -1: Full left
# 0: Center
# 1: Full right

# Throttle: [0, 1]
# 0: No throttle
# 1: Full throttle
```

### Action Mapping to CARLA

```python
def apply_action(vehicle, action):
    steer, throttle = action
    
    # Map from [-1, 1] to CARLA steering range
    steer = float(np.clip(steer, -1, 1))
    
    # Map from [0, 1] to CARLA throttle range
    throttle = float(np.clip(throttle, 0, 1))
    
    vehicle.apply_control(
        carla.VehicleControl(
            steer=steer,
            throttle=throttle,
            brake=0.0,
            hand_brake=False,
            reverse=False
        )
    )
```

## Reward Function

Xem chi tiết tại [11_Reward_Control.md](11_Reward_Control.md)

### Reward Components

```python
reward = (
    speed_factor *           # Khuyến khích tốc độ mục tiêu
    centering_factor *       # Giữ xe giữa lane
    angle_factor +           # Giữ heading đúng hướng
    heading_bonus -          # Bonus cho heading chính xác
    lane_deviation_penalty - # Phạt lệch lane
    collision_penalty -      # Phạt va chạm
    off_road_penalty -       # Phạt off-road
    stuck_penalty            # Phạt stuck
)
```

## Training Tips

### 1. Warm-up với VAE Pre-trained
- Đảm bảo VAE đã được training kỹ trước khi train PPO
- Kiểm tra reconstruction quality

### 2. Learning Rate Scheduling
```python
# Learning rate decay
scheduler = torch.optim.lr_scheduler.StepLR(
    agent.optimizer, 
    step_size=100000, 
    gamma=0.5
)
```

### 3. Reward Normalization
```python
# Normalize rewards trong buffer
rewards = np.array(rewards)
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

### 4. Gradient Clipping
```python
nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=0.5)
```

### 5. Early Stopping
```python
# Stop if average reward converges
if np.std(agent.episode_rewards) < 1.0:
    print("Converged! Stopping training.")
    break
```

## Monitoring Training

### TensorBoard Metrics

```bash
tensorboard --logdir=runs/
```

Theo dõi:
- `Loss/Policy`: Policy loss nên giảm
- `Loss/Value`: Value loss nên giảm
- `Loss/Entropy`: Entropy nên giảm dần
- `Episode/Reward`: Episode reward nên tăng
- `Stats/Clip Fraction`: Nên < 0.3

### Training Curves

```python
import matplotlib.pyplot as plt

def plot_training_curve(rewards, window=100):
    rewards = np.array(rewards)
    
    # Moving average
    ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Raw')
    plt.plot(ma, label=f'{window}-episode MA')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()
```

## Troubleshooting

### Lỗi: "Policy collapse"
- Giảm learning rate
- Tăng entropy bonus (c2)
- Giảm clip_epsilon

### Lỗi: "Value loss quá cao"
- Tăng c1 (value loss coefficient)
- Normalize rewards
- Check reward scaling

### Lỗi: "Training không hội tụ"
- Check VAE quality
- Adjust reward function
- Tăng update_frequency

## Next Steps

- [10_Dataset.md](10_Dataset.md) - Dataset collection
- [11_Reward_Control.md](11_Reward_Control.md) - Reward tuning
- [12_Checkpoint.md](12_Checkpoint.md) - Checkpoint management
- [13_Testing.md](13_Testing.md) - Testing agent
