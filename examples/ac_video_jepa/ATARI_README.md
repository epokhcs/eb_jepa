# ATARI Support for ac_video_jepa

This document describes how to use the ac_video_jepa framework with ATARI games, specifically Breakout.

## Overview

The ac_video_jepa framework has been extended to support ATARI environments with discrete action spaces. The implementation includes:

- **Environment abstraction layer**: Plugin-based architecture supporting multiple environments
- **Discrete action support**: Action embeddings, discrete inverse dynamics, categorical planning
- **ATARI environment**: Gymnasium wrapper with preprocessing (grayscale, resizing, normalization)
- **On-the-fly data generation**: Random policy trajectory collection
- **Backward compatibility**: Two Rooms environment continues to work unchanged

## Installation

### Basic Requirements

```bash
# Install base dependencies
pip install torch torchvision gymnasium

# Install ATARI support
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"  # Accept ROMs license
pip install ale-py

# Other dependencies
pip install einops omegaconf pyyaml tqdm pandas
```

### Verify Installation

```python
import gymnasium as gym

# Test ATARI environment creation
env = gym.make("ALE/Breakout-v5")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
env.close()
```

## Quick Start

### Training on ATARI Breakout

```bash
# Train ac_video_jepa on ATARI Breakout
python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml
```

### Key Configuration Parameters

```yaml
data:
  env_name: atari           # Environment type
  game_name: Breakout       # Specific ATARI game
  dobs: 1                   # Grayscale (1 channel)
  img_size: 84              # Standard ATARI resolution
  batch_size: 32
  sample_length: 17         # Training sequence length

model:
  action_embedding_dim: 64  # Discrete action embedding dimension

planning:
  action_type: discrete     # CRITICAL for ATARI
```

## Architecture Details

### Discrete Action Handling

**1. Action Encoder** ([eb_jepa/architectures.py](../../eb_jepa/architectures.py))
- `DiscreteActionEncoder`: Embeds action indices into continuous vectors
- Uses `nn.Embedding` to map discrete actions to learned representations

**2. RNN Predictor**
- Updated to accept optional `action_encoder` parameter
- Automatically encodes actions before passing to GRU

**3. Inverse Dynamics Model**
- `DiscreteInverseDynamicsModel`: Outputs logits for discrete actions
- Uses cross-entropy loss instead of MSE

**4. Planning**
- `CEMPlanner`: Updated to support categorical sampling
- Maintains distribution over discrete actions as logits
- Samples using temperature-controlled categorical distribution
- Updates parameters based on elite action frequencies

### Data Flow

```
ATARI Env → AtariDataset → TrajectoryBatch
                             ├─ states: [C, T, H, W]  (grayscale frames)
                             ├─ actions: [1, T]       (discrete indices)
                             └─ metadata: rewards, dones
                                          ↓
                             DiscreteActionEncoder
                                          ↓
                             RNNPredictor (with action encoding)
                                          ↓
                             Prediction + Regularization Losses
```

## Environment Details

### AtariEnv ([eb_jepa/datasets/atari/env.py](../../eb_jepa/datasets/atari/env.py))

**Features:**
- Wraps gymnasium ATARI environments
- Automatic grayscale conversion
- Resizing to 84x84
- Pixel normalization [0, 255] → [0, 1]
- Discrete action space (4 actions for Breakout)

**Action Space:**
- Breakout: 4 actions (NOOP, FIRE, RIGHT, LEFT)
- Actions represented as integer indices

**Observation Space:**
- Shape: (1, 84, 84) for grayscale
- Values: [0, 1] (normalized pixels)

### AtariDataset ([eb_jepa/datasets/atari/dataset.py](../../eb_jepa/datasets/atari/dataset.py))

**Data Generation:**
- On-the-fly trajectory collection
- Random policy (uniform action sampling)
- Automatic episode reset on termination
- Subsequence sampling for training batches

**Sample Format:**
```python
TrajectoryBatch(
    states=[C, T, H, W],      # Grayscale frames
    actions=[1, T],            # Discrete action indices
    metadata={
        "rewards": [T],        # Scalar rewards
        "dones": [T],          # Episode termination flags
    }
)
```

## Supported ATARI Games

The framework supports any ATARI game available in gymnasium. To use a different game:

```yaml
data:
  env_name: atari
  game_name: Pong  # Or: SpaceInvaders, MsPacman, etc.
```

**Note:** Different games have different action spaces. The framework automatically detects the action space size.

## Comparison: Two Rooms vs ATARI

| Aspect | Two Rooms | ATARI Breakout |
|--------|-----------|----------------|
| Action Space | Continuous 2D (velocity) | Discrete 4 actions |
| Observation | 2-channel (dot + wall) | 1-channel grayscale |
| Resolution | 65×65 | 84×84 |
| Goal | Position-based (reach target) | Score-based (maximize reward) |
| Evaluation | Euclidean distance | Score difference |
| Data Generation | Synthetic trajectories | Environment rollouts |

## Training Tips

### Hyperparameters

**For ATARI (recommended starting point):**
- `batch_size`: 32 (smaller than Two Rooms due to memory)
- `action_embedding_dim`: 64
- `sample_length`: 17 (same as Two Rooms)
- `lr`: 1e-4
- `idm_coeff`: 1.0 (important for discrete actions)

### Loss Coefficients

The regularization losses are critical for avoiding collapse:
- `var_coeff`: 16.0 (variance loss)
- `cov_coeff`: 8.0 (covariance loss)
- `sim_coeff`: 12.0 (time similarity)
- `idm_coeff`: 1.0 (inverse dynamics)

### Data Collection

Current implementation uses **random policy**. For better performance:
1. Train a simple DQN agent first
2. Collect trajectories with the trained agent
3. Use those trajectories for world model training

## Known Limitations

1. **Random Policy Data**: Current dataset uses random actions, which may not explore interesting states
2. **No Decoder**: ATARI planning adapter requires a decoder network (not implemented)
3. **Score-Based Goal Conditioning**: Unlike Two Rooms, ATARI doesn't have explicit goal states
4. **Memory Usage**: ATARI frames are larger than Two Rooms, requiring more memory

## Future Enhancements

- [ ] Pre-collected expert trajectories
- [ ] Decoder network for visualization
- [ ] Reward prediction head
- [ ] Value function for RL
- [ ] Frame stacking support
- [ ] Multi-game training

## Troubleshooting

### Import Error: gymnasium not found
```bash
pip install gymnasium
```

### Import Error: ale_py not found
```bash
pip install ale-py
pip install "gymnasium[accept-rom-license]"
```

### Action space mismatch
Make sure `action_type: discrete` is set in planning configuration.

### Memory issues
Reduce `batch_size` or `num_samples` in planning.

## References

- [Gymnasium ATARI Documentation](https://gymnasium.farama.org/environments/atari/)
- [ac_video_jepa Two Rooms Example](../README.md)
- [Planning with Discrete Actions](../../eb_jepa/planning.py)

## Contributing

To add support for a new environment:

1. Create environment class inheriting from `EnvBase`
2. Create dataset class inheriting from `DatasetBase`
3. Create config dataclass extending `DatasetConfigBase`
4. Register using `EnvironmentRegistry.register_env()`
5. Create `data_config.yaml` file
6. Update training configuration

See [eb_jepa/datasets/atari/](../../eb_jepa/datasets/atari/) for a complete example.
