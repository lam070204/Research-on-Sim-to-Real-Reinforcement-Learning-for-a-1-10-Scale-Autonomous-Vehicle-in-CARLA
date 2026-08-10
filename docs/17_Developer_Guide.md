# 17 Developer Guide

## Hướng dẫn cho Developer

Tài liệu hướng dẫn cho developers muốn đóng góp, mở rộng hoặc tùy chỉnh dự án.

## Mục lục

- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Quy trình phát triển](#quy-trình-phát-triển)
- [Code conventions](#code-conventions)
- [Testing guidelines](#testing-guidelines)
- [Documentation guidelines](#documentation-guidelines)
- [Contributing](#contributing)

---

## Kiến trúc hệ thống

### Tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                     Autonomous Driving System                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │   CARLA     │  │   Sensors   │  │   Environment       │  │
│  │  Simulation │──│  (Camera)   │──│   (state, reward)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                              │               │
│                                              ▼               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Deep Learning Models                    │    │
│  │  ┌──────────────┐  ┌──────────────┐                 │    │
│  │  │     VAE      │  │     PPO      │                 │    │
│  │  │  (Encoder)   │  │   (Policy)   │                 │    │
│  │  └──────────────┘  └──────────────┘                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                              │               │
│                                              ▼               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Vehicle Control                         │    │
│  │         (Steer, Throttle, Brake)                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
simulation/
├── environment.py ────────► sensors.py
│      │                        │
│      ▼                        ▼
│   reward.py              connection.py
│
networks/
├── on_policy/
│   └── ppo/
│       ├── agent.py ───────► ppo.py
│       └── utils.py
│
autoencoder/
├── vae.py ───────────────► encoder.py
│      │
│      ▼
│   decoder.py
```

### Data Flow

```
1. CARLA Simulation
   │
   ▼
2. Camera captures RGB image (160x80x3)
   │
   ▼
3. VAE Encoder processes image → latent (95,)
   │
   ▼
4. Navigation state extracted (5,)
   │
   ▼
5. Full state = latent + nav_state (100,)
   │
   ▼
6. PPO Policy → action (steer, throttle)
   │
   ▼
7. Vehicle control applied in CARLA
   │
   ▼
8. Reward calculated, episode continues or ends
```

---

## Quy trình phát triển

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/your-username/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning.git
cd Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8 mypy
```

### 2. Branch Strategy

```bash
# Main branches
main            # Production-ready code
develop         # Integration branch

# Feature branches (from develop)
feature/vae-improvement
feature/ppo-optimization
feature/new-town

# Bugfix branches
bugfix/camera-crash
bugfix/reward-calculation

# Release branches
release/v1.0.0
```

### 3. Commit Message Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Build/config changes

**Examples:**
```
feat(ppo): add gradient clipping to prevent explosion

Added max_grad_norm parameter to PPO agent
Clipped gradients to 0.5 to prevent instability

Closes #42

---

fix(sensors): fix camera image format conversion

Changed from BGRA to RGB conversion
Fixed issue #38

---

docs(readme): update installation instructions

Added Windows-specific installation steps
Updated CARLA version requirements
```

### 4. Pull Request Process

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added for new features

## Related Issues
Fixes #123
```

---

## Code Conventions

### Python Style Guide

```python
# Imports (standard library, third-party, local)
import os
import sys
from typing import Tuple, Optional

import numpy as np
import torch
import carla

from simulation.sensors import CameraSensor
from networks.ppo import PPOAgent

# Constants (UPPER_CASE)
DEFAULT_TOWN = 'Town07'
DEFAULT_PORT = 2000
MAX_SPEED = 30.0  # km/h

# Classes (CamelCase)
class CarlaEnvironment:
    """CARLA simulation environment."""
    
    def __init__(
        self,
        town: str = DEFAULT_TOWN,
        port: int = DEFAULT_PORT,
        timeout: float = 10.0
    ) -> None:
        self.town = town
        self.port = port
        self.timeout = timeout
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment and return initial state."""
        pass
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, dict]:
        """Execute action and return results."""
        pass

# Functions (snake_case)
def calculate_reward(
    speed: float,
    distance_from_center: float,
    heading_error: float
) -> float:
    """Calculate reward based on driving metrics.
    
    Args:
        speed: Current vehicle speed (km/h)
        distance_from_center: Distance from lane center (m)
        heading_error: Heading angle error (degrees)
    
    Returns:
        Reward value
    """
    speed_factor = np.clip(speed / MAX_SPEED, 0, 1)
    centering_factor = np.exp(-distance_from_center / 3.0)
    angle_factor = np.cos(np.radians(heading_error))
    
    return speed_factor * centering_factor * angle_factor

# Type hints
def process_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (80, 160)
) -> np.ndarray:
    """Process camera image."""
    pass

# Context managers
class Timer:
    """Context manager for timing code blocks."""
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"Elapsed: {elapsed:.3f}s")
```

### File Structure

```python
# Module docstring
"""VAE implementation for image encoding."""

# Imports
import torch
import torch.nn as nn

# Module constants
DEFAULT_LATENT_DIM = 95

# Classes
class VAE(nn.Module):
    """Variational Autoencoder."""
    pass

# Functions
def create_vae(latent_dim: int = DEFAULT_LATENT_DIM) -> VAE:
    """Create VAE model."""
    pass

# Main block
if __name__ == '__main__':
    # Test code
    vae = create_vae()
    print(vae)
```

### Error Handling

```python
class TrainingError(Exception):
    """Base exception for training errors."""
    pass

class ModelLoadingError(TrainingError):
    """Exception for model loading failures."""
    pass

def safe_load_checkpoint(path: str) -> dict:
    """Safely load checkpoint with error handling.
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
        
    Raises:
        ModelLoadingError: If checkpoint cannot be loaded
        FileNotFoundError: If file doesn't exist
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        return checkpoint
        
    except Exception as e:
        raise ModelLoadingError(f"Failed to load checkpoint: {e}")
```

---

## Testing Guidelines

### Unit Tests

```python
# tests/test_reward.py

import pytest
import numpy as np
from simulation.reward import calculate_reward

class TestRewardCalculation:
    """Test reward calculation functions."""
    
    def test_reward_range(self):
        """Test reward is in expected range."""
        reward = calculate_reward(
            speed=20.0,
            distance_from_center=0.0,
            heading_error=0.0
        )
        assert 0.0 <= reward <= 1.0
    
    def test_reward_perfect_driving(self):
        """Test reward for perfect driving."""
        reward = calculate_reward(
            speed=20.0,
            distance_from_center=0.0,
            heading_error=0.0
        )
        assert reward > 0.9
    
    def test_reward_off_road(self):
        """Test reward when off road."""
        reward = calculate_reward(
            speed=0.0,
            distance_from_center=5.0,
            heading_error=0.0
        )
        assert reward < 0.1
    
    def test_reward_stationary(self):
        """Test reward when stationary."""
        reward = calculate_reward(
            speed=0.0,
            distance_from_center=0.0,
            heading_error=0.0
        )
        assert reward < 0.5
    
    @pytest.mark.parametrize("speed,distance,angle", [
        (20.0, 0.0, 0.0),
        (15.0, 1.0, 5.0),
        (25.0, 0.5, 10.0),
        (0.0, 2.0, 15.0),
    ])
    def test_reward_various_conditions(self, speed, distance, angle):
        """Test reward with various driving conditions."""
        reward = calculate_reward(speed, distance, angle)
        assert isinstance(reward, float)
```

### Integration Tests

```python
# tests/test_environment.py

import pytest
import carla
from simulation.environment import CarlaEnvironment

@pytest.fixture
def carla_client():
    """Fixture for CARLA client."""
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    yield client

@pytest.fixture
def environment(carla_client):
    """Fixture for environment."""
    env = CarlaEnvironment(client=carla_client)
    yield env
    env.close()

class TestEnvironment:
    """Test environment integration."""
    
    def test_reset(self, environment):
        """Test environment reset."""
        state = environment.reset()
        assert isinstance(state, tuple)
        assert len(state) == 2
    
    def test_step(self, environment):
        """Test environment step."""
        environment.reset()
        action = np.array([0.0, 0.5])  # steer, throttle
        next_state, reward, done, info = environment.step(action)
        
        assert isinstance(next_state, tuple)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_episode(self, environment):
        """Test full episode."""
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < 100:
            action = environment.action_space.sample()
            next_state, reward, done, info = environment.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        assert steps > 0
        assert total_reward != 0
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_reward.py

# Run specific test class
pytest tests/test_reward.py::TestRewardCalculation

# Run with coverage
pytest --cov=simulation --cov=networks tests/

# Run with verbose output
pytest -v tests/

# Run tests matching pattern
pytest -k "test_reward" tests/
```

---

## Documentation Guidelines

### Docstring Format

```python
def train_agent(
    agent: PPOAgent,
    environment: CarlaEnvironment,
    num_episodes: int,
    checkpoint_dir: str,
    log_interval: int = 10
) -> TrainingResult:
    """Train PPO agent using environment.
    
    This function implements the main training loop for the PPO agent.
    It handles episode execution, reward collection, and policy updates.
    
    Args:
        agent: PPO agent to train
        environment: CARLA environment for training
        num_episodes: Number of episodes to train
        checkpoint_dir: Directory to save checkpoints
        log_interval: Interval for logging (default: 10)
    
    Returns:
        TrainingResult object containing:
            - rewards: List of episode rewards
            - losses: Training losses per update
            - agent: Trained agent
    
    Raises:
        TrainingError: If training fails
        ValueError: If parameters are invalid
    
    Example:
        >>> agent = PPOAgent(latent_dim=95, nav_dim=5, action_dim=2)
        >>> env = CarlaEnvironment(town='Town07')
        >>> result = train_agent(agent, env, num_episodes=1000)
        >>> print(f"Average reward: {np.mean(result.rewards[-100:]):.2f}")
    
    Note:
        - Checkpoints are saved every 100 episodes
        - Training uses GPU if available
        - Early stopping if reward > 0.9 for 50 consecutive episodes
    """
    pass
```

### README Sections

```markdown
# Module Name

Brief description (1-2 sentences)

## Installation

```bash
pip install module-name
```

## Quick Start

```python
from module import ClassName

obj = ClassName()
result = obj.method()
```

## API Reference

### ClassName

#### `method_name(args)`

Description

**Args:**
- `arg1`: Description

**Returns:**
- Description

**Example:**
```python
obj.method_name(value)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| param1 | value | Description |

## Examples

See `examples/` directory for more examples.

## Troubleshooting

Common issues and solutions.

## References

- Paper links
- Documentation links
```

---

## Contributing

### How to Contribute

1. **Fork the repository**
```bash
git fork https://github.com/original/repo.git
```

2. **Create feature branch**
```bash
git checkout -b feature/your-feature
```

3. **Make changes and commit**
```bash
git add .
git commit -m "feat(scope): description"
```

4. **Push and create PR**
```bash
git push origin feature/your-feature
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No unnecessary dependencies
- [ ] Error handling implemented
- [ ] Type hints added
- [ ] Comments for complex logic

### Areas for Contribution

**High Priority:**
- [ ] DDQN implementation improvements
- [ ] Reward function tuning
- [ ] Training stability improvements
- [ ] Documentation updates

**Medium Priority:**
- [ ] Additional towns support
- [ ] Weather robustness training
- [ ] Multi-agent scenarios
- [ ] Real-world transfer improvements

**Low Priority:**
- [ ] UI improvements
- [ ] Additional sensors
- [ ] Performance optimizations
- [ ] Example scripts

---

## Development Tools

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.7, 3.8, 3.9]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest --cov=. tests/
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Next Steps

- [01_Project_Overview.md](01_Project_Overview.md) - Project overview
- [16_API.md](16_API.md) - API reference
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
