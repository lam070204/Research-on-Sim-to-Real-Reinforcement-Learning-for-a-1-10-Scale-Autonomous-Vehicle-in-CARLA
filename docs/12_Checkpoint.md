# 12 Checkpoint

## Checkpoint Management

Tài liệu về quản lý checkpoint cho models trong dự án.

## Tổng quan

Checkpoint là các file lưu trữ weights và trạng thái của models trong quá trình training. Việc quản lý checkpoint tốt giúp:
- Resume training sau khi bị gián đoạn
- Load models đã trained để testing/deployment
- Backup các versions khác nhau
- Compare performance giữa các versions

## Cấu trúc Checkpoints

```
checkpoints/
├── PPO/
│   ├── ppo_episode_0.pth
│   ├── ppo_episode_10000.pth
│   ├── ppo_episode_20000.pth
│   ├── ...
│   └── ppo_episode_1000000.pth
└── DDQN/
    ├── ddqn_episode_0.pth
    ├── ddqn_episode_10000.pth
    └── ...

preTrained_models/
├── ppo/
│   └── best_ppo.pth
└── ddqn/
    └── best_ddqn.pth
```

## Checkpoint Format

### PPO Checkpoint

```python
import torch

def save_ppo_checkpoint(agent, episode, path):
    """
    Save PPO checkpoint.
    
    Args:
        agent: PPOAgent instance
        episode: Current episode number
        path: Path to save checkpoint
    """
    checkpoint = {
        'episode': episode,
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode_rewards': list(agent.episode_rewards),
        'episode_lengths': list(agent.episode_lengths),
        'hyperparameters': {
            'latent_dim': 95,
            'nav_dim': 5,
            'action_dim': 2,
            'hidden_dim': 256,
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'c1': 0.5,
            'c2': 0.01
        }
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_ppo_checkpoint(agent, path, device='cuda'):
    """
    Load PPO checkpoint.
    
    Args:
        agent: PPOAgent instance
        path: Path to checkpoint
        device: Device to load to
    
    Returns:
        episode: Episode number when checkpoint was saved
    """
    checkpoint = torch.load(path, map_location=device)
    
    agent.policy.load_state_dict(checkpoint['policy_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
    agent.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=100)
    
    return checkpoint['episode']
```

### VAE Checkpoint

```python
def save_vae_checkpoint(vae, epoch, optimizer, path):
    """
    Save VAE checkpoint.
    
    Args:
        vae: VAE model
        epoch: Current epoch number
        optimizer: Optimizer
        path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': vae.loss,
        'hyperparameters': {
            'input_dim': 38400,
            'latent_dim': 95,
            'hidden_dim': 512
        }
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_vae_checkpoint(vae, path, device='cuda'):
    """
    Load VAE checkpoint.
    
    Args:
        vae: VAE model
        path: Path to checkpoint
        device: Device to load to
    
    Returns:
        epoch: Epoch number when checkpoint was saved
    """
    checkpoint = torch.load(path, map_location=device)
    
    vae.load_state_dict(checkpoint['model_state_dict'])
    
    return checkpoint['epoch']
```

## Checkpoint Strategies

### Strategy 1: Periodic Checkpoints

Save checkpoint sau mỗi N episodes:

```python
save_frequency = 10000

for episode in range(num_episodes):
    # Training loop
    ...
    
    # Save checkpoint
    if episode % save_frequency == 0:
        agent.save(f'checkpoints/PPO/ppo_episode_{episode}.pth')
```

### Strategy 2: Best Model Checkpoints

Save checkpoint khi đạt best performance:

```python
best_reward = -float('inf')
best_episode = 0

for episode in range(num_episodes):
    # Training loop
    avg_reward = np.mean(agent.episode_rewards)
    
    # Save best model
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_episode = episode
        agent.save('checkpoints/PPO/best_ppo.pth')
        print(f"New best model at episode {episode} with reward {best_reward:.2f}")
```

### Strategy 3: Last N Checkpoints

Chỉ giữ lại N checkpoints gần nhất:

```python
import os
import glob

def keep_last_n_checkpoints(checkpoint_dir, n=5):
    """
    Keep only the last N checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        n: Number of checkpoints to keep
    """
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))
    
    # Remove old checkpoints
    for checkpoint in checkpoints[:-n]:
        os.remove(checkpoint)
        print(f"Removed old checkpoint: {checkpoint}")
```

### Strategy 4: Checkpoint with Metrics

Save checkpoint kèm theo metrics:

```python
import json

def save_checkpoint_with_metrics(agent, episode, metrics, path):
    checkpoint = {
        'episode': episode,
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'metrics': metrics
    }
    
    torch.save(checkpoint, path)
    
    # Save metrics separately for easy viewing
    metrics_path = path.replace('.pth', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
```

## Resume Training

### Resume PPO Training

```python
def resume_ppo_training(
    checkpoint_path,
    num_episodes=1000000,
    save_frequency=10000
):
    # Initialize agent
    agent = PPOAgent(
        latent_dim=95,
        nav_dim=5,
        action_dim=2,
        hidden_dim=256,
        device='cuda'
    )
    
    # Load checkpoint
    start_episode = load_ppo_checkpoint(agent, checkpoint_path)
    print(f"Resumed from episode {start_episode}")
    
    # Initialize environment
    env = CarlaEnvironmentRGB(town='Town07')
    encoder = EncodeStateRGBV2(latent_dim=95, device='cuda')
    encoder.load_weights('autoencoder_rgb/model/vae_rgb_epoch_100.pth')
    
    # TensorBoard
    writer = SummaryWriter(log_dir='runs/PPO_resumed')
    
    buffer = []
    total_steps = start_episode * 1000  # Assuming 1000 steps per episode
    
    for episode in range(start_episode, num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(1000):
            # Training steps
            ...
            
            episode_reward += reward
            total_steps += 1
            
            if done:
                break
        
        # Log and save
        if episode % 100 == 0:
            writer.add_scalar('Episode/Reward', episode_reward, episode)
        
        if episode % save_frequency == 0:
            agent.save(f'checkpoints/PPO/ppo_episode_{episode}.pth')
    
    writer.close()
```

## Checkpoint Utilities

### List Available Checkpoints

```python
import os

def list_checkpoints(checkpoint_dir='checkpoints/PPO'):
    """
    List all available checkpoints.
    
    Args:
        checkpoint_dir: Directory to search
    
    Returns:
        List of checkpoint paths
    """
    checkpoints = []
    
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.pth'):
            path = os.path.join(checkpoint_dir, f)
            checkpoint = torch.load(path, map_location='cpu')
            
            if 'episode' in checkpoint:
                episode = checkpoint['episode']
                reward = np.mean(checkpoint.get('episode_rewards', [0]))
                checkpoints.append((path, episode, reward))
    
    # Sort by episode
    checkpoints.sort(key=lambda x: x[1])
    
    print("Available Checkpoints:")
    print("-" * 60)
    for path, episode, reward in checkpoints:
        print(f"Episode {episode:6d} | Reward: {reward:6.2f} | {path}")
    
    return checkpoints
```

### Compare Checkpoints

```python
def compare_checkpoints(checkpoint_paths):
    """
    Compare multiple checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint paths
    """
    import pandas as pd
    
    data = []
    
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location='cpu')
        
        data.append({
            'Checkpoint': os.path.basename(path),
            'Episode': checkpoint.get('episode', 0),
            'Avg Reward': np.mean(checkpoint.get('episode_rewards', [0])),
            'Std Reward': np.std(checkpoint.get('episode_rewards', [0])),
            'Avg Length': np.mean(checkpoint.get('episode_lengths', [0]))
        })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    return df
```

### Convert Checkpoint Format

```python
def convert_checkpoint_format(old_path, new_path):
    """
    Convert checkpoint to new format.
    
    Args:
        old_path: Path to old checkpoint
        new_path: Path to save new checkpoint
    """
    old_checkpoint = torch.load(old_path, map_location='cpu')
    
    # Create new format
    new_checkpoint = {
        'episode': old_checkpoint.get('episode', 0),
        'policy_state_dict': old_checkpoint.get('policy_state_dict', old_checkpoint),
        'optimizer_state_dict': old_checkpoint.get('optimizer_state_dict', {}),
        'episode_rewards': old_checkpoint.get('episode_rewards', []),
        'episode_lengths': old_checkpoint.get('episode_lengths', []),
        'timestamp': time.time()
    }
    
    torch.save(new_checkpoint, new_path)
    print(f"Converted {old_path} to {new_path}")
```

## Checkpoint Best Practices

### 1. Regular Backups
```python
# Backup to cloud storage
import shutil

def backup_checkpoint(checkpoint_path, backup_dir):
    shutil.copy2(checkpoint_path, backup_dir)
```

### 2. Checkpoint Validation
```python
def validate_checkpoint(path, device='cuda'):
    """Validate checkpoint can be loaded."""
    try:
        checkpoint = torch.load(path, map_location=device)
        assert 'policy_state_dict' in checkpoint or 'model_state_dict' in checkpoint
        print(f"✓ Valid checkpoint: {path}")
        return True
    except Exception as e:
        print(f"✗ Invalid checkpoint: {path} - {e}")
        return False
```

### 3. Checkpoint Compression
```python
def compress_checkpoint(checkpoint_path):
    """Compress checkpoint to save space."""
    import gzip
    
    with open(checkpoint_path, 'rb') as f_in:
        with gzip.open(checkpoint_path + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    
    print(f"Compressed {checkpoint_path}")
```

### 4. Checkpoint Documentation
```python
def save_checkpoint_info(checkpoint_path, info):
    """Save additional info about checkpoint."""
    info_path = checkpoint_path.replace('.pth', '_info.txt')
    
    with open(info_path, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Created: {time.ctime()}\n")
        f.write(f"Info: {info}\n")
```

## Troubleshooting

### Issue: Checkpoint too large
- Save only state_dict instead of full model
- Use compression
- Remove unnecessary data from checkpoint

### Issue: Cannot load checkpoint
- Check device (CPU vs GPU)
- Check PyTorch version compatibility
- Verify checkpoint format

### Issue: Out of memory when loading
- Load to CPU first: `torch.load(path, map_location='cpu')`
- Use checkpoint compression
- Clear cache: `torch.cuda.empty_cache()`

## Next Steps

- [13_Testing.md](13_Testing.md) - Testing agent với checkpoints
- [14_Deployment.md](14_Deployment.md) - Deployment với trained models
- [11_Reward_Control.md](11_Reward_Control.md) - Reward tuning
