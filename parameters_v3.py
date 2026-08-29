"""
Canonical project hyperparameters.

GĐ15 keeps legacy DQN constants for compatibility, but PPO V3 uses the
explicit V3 section below.
"""

MODEL_LOAD = False
SEED = 0

# Shared / legacy.
BATCH_SIZE = 1
IM_WIDTH = 160
IM_HEIGHT = 80
GAMMA = 0.99
MEMORY_SIZE = 5000
EPISODES = 1000

# VAE bottleneck.
LATENT_DIM = 95

# ----------------------------------------------------------------
# Dueling DQN legacy parameters.
# ----------------------------------------------------------------
DQN_LEARNING_RATE = 0.0001
EPSILON = 1.00
EPSILON_END = 0.05
EPSILON_DECREMENT = 0.00001

REPLACE_NETWORK = 5
DQN_CHECKPOINT_DIR = "preTrained_models/ddqn"
MODEL_ONLINE = "carla_dueling_dqn_online.pth"
MODEL_TARGET = "carla_dueling_dqn_target.pth"

# ----------------------------------------------------------------
# PPO V3.
# ----------------------------------------------------------------
TOTAL_TIMESTEPS = 2_000_000
TEST_TIMESTEPS = 50_000

PPO_LEARNING_RATE = 1e-4
POLICY_CLIP = 0.2

# Gaussian std in RAW action space.
ACTION_STD_INIT = 0.20
PPO_ACTION_STD_MIN = 0.05
PPO_ACTION_STD_DECAY = 0.01
PPO_ACTION_STD_DECAY_FREQ = 100_000

# 2048 policy steps ~= 68.3 s at 30 Hz.
PPO_ROLLOUT_STEPS = 2048

# Environment has its own simulation-time termination.
PPO_MAX_EPISODE_SECONDS = 60.0

PPO_CHECKPOINT_EVERY_STEPS = 50_000
PPO_CHECKPOINT_DIR = "preTrained_models/ppo/"

# Curriculum is indexed by fraction of TOTAL_TIMESTEPS.
# Speed is real-equivalent m/s.
PPO_CURRICULUM = (
    (0.00, 0.40),
    (0.25, 0.60),
    (0.50, 0.80),
    (0.75, 1.00),
)

# Legacy name retained so old scripts importing it do not crash.
# Clean V3 trainer does not use this value.
EPISODE_LENGTH = 7500
