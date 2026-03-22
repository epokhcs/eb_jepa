# Training Guide

Complete guide to training the JEPA world model on ATARI Breakout.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Reference](#configuration-reference)
3. [Data Collection Strategies](#data-collection-strategies)
4. [Training Commands](#training-commands)
5. [Monitoring Training](#monitoring-training)
6. [Expected Results](#expected-results)
7. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Development (Fast Test)
```bash
uv run python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml \
  --optim.epochs 1 \
  --data.size 1000
```

### Production (Full Training with Random Policy Data)
```bash
uv run python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml
```

### Production (With Systematic Paddle Sweep - Recommended)
```bash
# 1. Generate systematic dataset first
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --paddle_positions 20 \
  --episodes_per_position 3 \
  --output_dir data/systematic_sweep

# 2. Train with systematic dataset
uv run python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train_atari_systematic.yaml
```

---

## Configuration Reference

### Standard Training Configuration

See `examples/ac_video_jepa/cfgs/train_atari.yaml` for complete configuration.

**Key parameters:**
```yaml
data:
  env_name: atari
  game_name: Breakout
  batch_size: 32        # Adjust based on GPU memory
  size: 10000           # Number of trajectories
  val_size: 1000

model:
  dobs: 1               # Grayscale
  encoder_architecture: impala
  reward_prediction: true  # Enable reward head
  reward_head_hidden_dim: 256
  reward_loss_coeff: 1.0

optim:
  epochs: 5             # Quick: 5 / Production: 50
  lr: 0.0001
  grad_clip_enc: 2.0
  grad_clip_pred: 2.0
```

### Systematic Dataset Configuration

See `examples/ac_video_jepa/cfgs/train_atari_systematic.yaml`.

**Key differences:**
```yaml
data:
  size: 60              # Pre-collected trajectories
  val_size: 12
  use_systematic_dataset: true
  systematic_dataset_path: data/systematic_sweep/breakout_systematic_paddle_sweep.pkl

optim:
  epochs: 10            # More epochs with better data coverage
```

---

## Data Collection Strategies

### Random Policy Data Collection (Default)

The standard approach uses random policy to collect trajectories:
- **Pros**: Fast data collection, simple implementation
- **Cons**: Uneven paddle position coverage, biased initial conditions
- **Coverage**: ~60% of paddle positions (clustered around center)

**When to use:**
- Quick prototyping
- Initial testing
- Limited compute resources

### Systematic Paddle Sweep (Recommended)

Systematically varies paddle position before firing to ensure uniform coverage of initial conditions.

#### The Problem with Random Policy

During standard random policy data collection, the paddle position before firing the ball is not systematically varied. This leads to:
- **Limited initial condition coverage**: Model only sees trajectories from a narrow range of paddle starting positions
- **Biased learning**: World model may not generalize well to unseen paddle positions
- **Poor prediction quality**: Without diverse initial conditions, model struggles to predict future states accurately

#### The Solution: Systematic Sweep

The systematic paddle sweep ensures uniform coverage of all possible initial conditions:

1. **Reset game** → paddle at random position
2. **Move paddle all the way LEFT** (50 steps)
3. **FIRE ball** from leftmost position
4. **Collect trajectory** (200 frames)
5. **Reset game**
6. **Move paddle LEFT** then **RIGHT by 1 step**
7. **FIRE ball** from position LEFT+1
8. **Collect trajectory**
9. Repeat steps 5-8, incrementing RIGHT steps until we cover the full paddle range

#### Expected Benefits

1. **Better Initial Condition Coverage**
   - Uniform sampling of paddle positions ensures model sees diverse starting states
   - Reduces bias toward specific paddle positions

2. **Improved Generalization**
   - World model learns from full range of possible trajectories
   - Better prediction quality across all paddle positions

3. **Enhanced Planning Performance**
   - MPPI/CEM planners can leverage better world model predictions
   - More confident action selection across diverse states

4. **Reduced Sample Complexity**
   - Systematic exploration is more efficient than random exploration
   - Fewer trajectories needed to cover the full state space

#### Comparison: Random vs Systematic

| Aspect | Random Policy | Systematic Sweep |
|--------|---------------|------------------|
| **Paddle Coverage** | Uneven, biased (~60%) | Uniform, complete (100%) |
| **Initial Conditions** | Random, clustered | Systematic, diverse |
| **Sample Efficiency** | Low (redundant data) | High (targeted coverage) |
| **World Model Quality** | Moderate | Better |
| **Planning Performance** | Moderate | Improved |
| **Data Collection Time** | Fast | Moderate |

---

## Paddle Policies (During Gameplay)

**CRITICAL CHOICE:** After firing the ball, what should the paddle do during trajectory collection?

The paddle policy determines **action diversity** and whether the model learns **action-conditioned dynamics**.

### Policy Options

#### 1. Static Policy (Default - NOT RECOMMENDED)

**Behavior:** Paddle stays still (NOOP only) after firing

```bash
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --paddle_policy static \
  --episodes_per_position 3
```

**Coverage:**
- Paddle unique positions: ~140 pixels
- Ball unique positions: ~652 pixels
- Action diversity: NOOP only (100%)
- Rewards: 0.0 (ball falls off screen)

**Pros:**
- ✅ Simple, predictable
- ✅ Learns passive ball physics

**Cons:**
- ❌ **Paddle action has NO effect on future states**
- ❌ JEPA doesn't learn action-conditioned dynamics
- ❌ Planning can't optimize paddle movements
- ❌ Low rewards (ball falls without paddle catching it)

**Use only if:** You explicitly want to learn passive ball physics without paddle control

#### 2. Tracking Policy (RECOMMENDED ✅)

**Behavior:** Paddle actively tracks the ball (moves LEFT/RIGHT to stay under it)

```bash
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --paddle_policy tracking \
  --episodes_per_position 3
```

**Coverage:**
- Paddle unique positions: **608 pixels (4.3x improvement!)**
- Ball unique positions: **1,363 pixels (2.1x improvement!)**
- Action diversity: LEFT/RIGHT/NOOP (realistic gameplay)
- Rewards: 1.0 (actually hits bricks!)

**Pros:**
- ✅ **Rich action diversity**: paddle moves based on ball position
- ✅ **Action-conditioned learning**: JEPA learns how paddle actions affect ball trajectory
- ✅ **Realistic gameplay**: paddle tries to catch ball
- ✅ **Higher rewards**: Successfully hits bricks (1.0 vs 0.0)
- ✅ **Better for planning**: learns paddle-ball interaction dynamics

**Cons:**
- ⚠️ May be "too good" - less exploration of bad paddle movements

**Use when:** Training for JEPA + planning applications (MPPI/CEM)

#### 3. Mixed Policy (Balanced Approach)

**Behavior:** 70% tracking + 30% random exploration

```bash
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --paddle_policy mixed \
  --episodes_per_position 3
```

**Coverage:**
- Paddle unique positions: **416 pixels (3.0x improvement)**
- Ball unique positions: **1,114 pixels (1.7x improvement)**
- Action diversity: LEFT/RIGHT/NOOP (balanced)
- Rewards: 1.0 (still catches ball sometimes)

**Pros:**
- ✅ Combines tracking with exploratory noise
- ✅ Learns from both good and bad paddle movements
- ✅ Good action diversity
- ✅ Maintains reasonable rewards

**Cons:**
- ⚠️ Slightly less coverage than pure tracking

**Use when:** You want balanced exploration with realistic gameplay

### Paddle Policy Comparison

| Policy | Paddle Coverage | Ball Coverage | Action Diversity | Rewards | Recommended |
|--------|----------------|---------------|------------------|---------|-------------|
| **Static** | 140 px (1.0x) | 652 px (1.0x) | NOOP only | 0.0 | ❌ No |
| **Mixed** | 416 px (3.0x) | 1,114 px (1.7x) | LEFT/RIGHT/NOOP | 1.0 | ✅ Yes |
| **Tracking** | 608 px (4.3x) | 1,363 px (2.1x) | LEFT/RIGHT/NOOP | 1.0 | ✅✅ **Best** |

### Impact on JEPA Planning

**With Static Policy:**
- **Planning variables**: Only initial paddle position (where to fire from)
- **Optimization**: One-shot decision
- **Learned dynamics**: Ball bounces, no paddle interaction
- **Use case**: ❌ Limited - can't control paddle during gameplay

**With Tracking/Mixed Policy:**
- **Planning variables**: Initial position + paddle movements over time
- **Optimization**: Dynamic trajectory planning (where to position paddle each frame)
- **Learned dynamics**: Ball bounces + paddle-ball collisions + catching behavior
- **Use case**: ✅ Full gameplay - can plan paddle movements to maximize score

### Action Distribution Analysis

Expected action distribution during trajectory collection:

**Static Policy:**
```
NOOP: 100%
FIRE: 0%
RIGHT: 0%
LEFT: 0%
```

**Tracking Policy:**
```
NOOP: ~40-50%
LEFT: ~25-30%
RIGHT: ~25-30%
FIRE: ~1% (only at start)
```

**Mixed Policy:**
```
NOOP: ~40-50%
LEFT: ~20-25%
RIGHT: ~20-25%
FIRE: ~1% (only at start)
```

### Heatmap Visualization

After dataset generation, inspect the heatmaps to verify coverage:

**Files generated:**
- `breakout_heatmaps.png` - Visual heatmap showing paddle and ball positions
- `breakout_heatmaps.npz` - Raw heatmap arrays for analysis

**What to look for:**
- **Paddle heatmap**: Horizontal band at bottom with wide coverage (tracking/mixed show 4x more coverage)
- **Ball heatmap**: Dense coverage across playfield (tracking/mixed show 2x more unique positions)

**Ideal pattern (tracking policy):**
- Paddle: Uniform coverage across full width (608 unique positions)
- Ball: High coverage throughout playfield (1,363 unique positions)
- No gaps or heavy clustering

**Poor pattern (static policy):**
- Paddle: Narrow coverage, many gaps (140 unique positions only)
- Ball: Limited exploration (652 unique positions)
- Clustered at initial position

---

## Paddle Policy Recommendation

**For JEPA + Planning applications, ALWAYS use `--paddle_policy tracking` or `--paddle_policy mixed`.**

The dramatic increase in paddle position coverage (4.3x) and action diversity means your world model will learn **action-conditioned dynamics** instead of just passive physics. This is essential for planning algorithms like MPPI/CEM to optimize paddle movements during gameplay.

**Static policy should only be used if you explicitly want to learn passive ball physics without paddle control.**

---

## Training Commands

### Step 1: Generate Systematic Sweep Dataset

Use the systematic sweep script to generate high-quality training data:

```bash
# RECOMMENDED: Tracking policy for action-conditioned learning
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --game Breakout \
  --paddle_policy tracking \
  --episodes_per_position 3 \
  --trajectory_length 200 \
  --output_dir data/systematic_sweep

# Alternative: Mixed policy (70% tracking + 30% random)
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --game Breakout \
  --paddle_policy mixed \
  --episodes_per_position 3 \
  --trajectory_length 200 \
  --output_dir data/systematic_sweep

# Testing: Quick test with fewer positions
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --game Breakout \
  --paddle_policy tracking \
  --paddle_positions 10 \
  --episodes_per_position 2 \
  --trajectory_length 100 \
  --output_dir data/systematic_sweep_test
```

**⚠️ Important: Paddle Position Coverage**

Breakout's paddle can only occupy **16 discrete positions** (quantized movement):
- Left edge: x = 8 pixels
- Right edge: x = 144 pixels
- Step size: 9 pixels per RIGHT action
- Total unique positions: 16

**Always omit `--paddle_positions` to auto-detect all 16 positions.** Manually specifying fewer positions (e.g., 10) will result in **incomplete coverage** with gaps in the data.

**Parameters:**
- `--game`: ATARI game name (default: Breakout)
- `--paddle_policy`: Paddle behavior during gameplay (default: static)
  - `tracking`: ✅ **Recommended** - Paddle tracks ball (4.3x coverage)
  - `mixed`: ✅ Good - 70% tracking + 30% random (3.0x coverage)
  - `static`: ❌ Not recommended - Paddle stays still (1.0x coverage)
- `--paddle_positions`: Number of paddle positions (default: auto-detect all 16). **Leave unspecified for complete coverage!**
- `--episodes_per_position`: Episodes to collect at each position (default: 3)
- `--trajectory_length`: Maximum frames per trajectory (default: 200)
- `--output_dir`: Output directory for dataset (default: data/systematic_sweep)

**Expected output (auto-detect mode with tracking policy):**
```
Determining paddle movement range...
Paddle range: 8 (left) to 144 (right)
Total range: 136 pixels
Paddle step size: 9 pixels per RIGHT action
Auto-detected paddle positions: 16
Paddle policy: tracking

Total trajectories collected: 48  (16 positions × 3 episodes)
Total frames: 9,600
Paddle Position Coverage: 16/16 unique positions sampled (100%)
File size: ~32 MB

Paddle Policy Statistics:
  Action distribution: NOOP: 45%, LEFT: 27%, RIGHT: 27%, FIRE: 1%
  Paddle unique positions: 608 (4.3x better than static)
  Ball unique positions: 1,363 (2.1x better than static)

Heatmap Statistics:
  Paddle total visits: 768,000
  Paddle max visits (hottest pixel): ~3,000
  Paddle unique positions: ~1,000
  Ball total visits: ~32,000
  Ball max visits (hottest pixel): ~60
  Ball unique positions: ~8,000
```

**Generated files:**
- `breakout_systematic_paddle_sweep.pkl` - Training dataset
- `breakout_heatmaps.png` - Visualization of paddle and ball coverage
- `breakout_heatmaps.npz` - Raw heatmap arrays for analysis

**For more thorough coverage (optional):**
```bash
uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
  --paddle_policy tracking \
  --paddle_positions 40 \
  --episodes_per_position 5 \
  --trajectory_length 200
```

This generates 200 trajectories (40 positions × 5 episodes) with tracking policy for even better coverage and action-conditioned learning.

### Step 2: Train World Model

#### Option A: Train with Systematic Dataset (Recommended)

```bash
uv run python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train_atari_systematic.yaml
```

#### Option B: Train with Random Policy Data

```bash
uv run python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml
```

### Step 3: Evaluate Planning Performance

After training, test planning with the learned world model:

```bash
uv run python -m examples.ac_video_jepa.test_planning_with_rewards \
  --checkpoint logs/training/atari_systematic/checkpoints/checkpoint_epoch_10.pth
```

Expected: Higher MPPI scores due to better world model (if trained with systematic data).

---

## Monitoring Training

### Loss Components

During training, monitor these loss components:

1. **Prediction MSE** (`pred_loss`): MSE between predicted and target latent states
   - Target: ~0.25 after 5 epochs
   - Lower is better (but too low may indicate overfitting)

2. **Reward MSE** (`reward_loss`): MSE between predicted and actual rewards
   - Target: ~0.23 after 5 epochs
   - Critical for planning with reward prediction

3. **VICReg Regularization**:
   - `var_loss`: Variance loss (prevents collapse)
   - `cov_loss`: Covariance loss (decorrelates features)
   - `sim_loss`: Similarity loss (temporal consistency)

4. **IDM Loss** (`idm_loss`): Inverse dynamics model prediction accuracy
   - Helps learn action-conditional dynamics

### Weights & Biases Integration

Training automatically logs to W&B if configured:

```yaml
logging:
  log_wandb: true
  log_every: 10
  exp_suffix: atari_systematic
```

Monitor at: https://wandb.ai/

### Checkpoint Files

Models are saved to:
```
logs/training/{exp_name}/checkpoints/
  ├── checkpoint_epoch_1.pth
  ├── checkpoint_epoch_2.pth
  └── ...
```

Each checkpoint (~260MB) contains:
- `model_state_dict`: JEPA encoder + predictor weights
- `reward_head_state_dict`: Reward prediction head weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Current epoch number

---

## Expected Results

### After 5 Epochs (~20 min on M4 Mac / ~10 min on GPU)

**Random Policy Data:**
- Prediction MSE: ~0.25
- Reward MSE: ~0.23
- Planning score (MPPI): 5-10 points per episode

**Systematic Sweep Data:**
- Prediction MSE: ~0.20 (expected improvement)
- Reward MSE: ~0.20 (expected improvement)
- Planning score (MPPI): 10-15 points per episode (expected improvement)
- Better generalization across paddle positions

### Dataset Statistics

**Random Policy Dataset:**
- Size: 10,000 trajectories
- Frames: ~2M total
- Paddle coverage: ~60% (uneven distribution)
- Action diversity: Random
- Storage: ~16 GB

**Systematic Sweep Dataset (Tracking Policy - RECOMMENDED):**
- Size: 48 trajectories (16 positions × 3 episodes)
- Frames: 9,600 total
- Paddle coverage: 100% (uniform distribution)
- Paddle unique positions: **608 pixels (4.3x vs static)**
- Ball unique positions: **1,363 pixels (2.1x vs static)**
- Action diversity: **NOOP: 45%, LEFT: 27%, RIGHT: 27%, FIRE: 1%**
- Rewards: 1.0 (hits bricks!)
- Storage: ~32 MB
- Efficiency: ~0.67 MB per trajectory

**Systematic Sweep Dataset (Static Policy - NOT RECOMMENDED):**
- Size: 48 trajectories (16 positions × 3 episodes)
- Frames: 9,600 total
- Paddle coverage: 100% (initial positions only)
- Paddle unique positions: **140 pixels**
- Ball unique positions: **652 pixels**
- Action diversity: **NOOP: 100%** (no paddle control)
- Rewards: 0.0 (ball falls off screen)
- Storage: ~32 MB
- **Problem:** Model learns passive physics, not action-conditioned dynamics

---

## Advanced Topics

### Dataset Format

The systematic sweep dataset is saved as a pickle file:

```python
{
    'trajectories': [
        {
            'observations': np.array,  # [T, H, W, C] RGB frames (210×160×3)
            'actions': np.array,       # [T] discrete actions (0-3)
            'rewards': np.array,       # [T] scalar rewards
            'length': int,             # Trajectory length
            'paddle_start_position': int,  # Paddle x-coord before firing
            'position_index': int,     # Sweep position index (0 to N-1)
        },
        ...
    ],
    'paddle_position_map': [0, 0, 0, 1, 1, 1, ...],  # Position index per trajectory
    'metadata': {
        'game': 'Breakout',
        'paddle_positions': 20,
        'episodes_per_position': 3,
        'trajectory_length': 200,
        'total_trajectories': 60,
    }
}
```

After loading by `SystematicAtariDataset`:
- Converted to grayscale
- Resized to 84×84
- Normalized to [0, 1]
- Format: [1, T, 84, 84] (grayscale channel, time, height, width)

### Paddle Position Sweep Strategy

The systematic sweep ensures uniform coverage by:
1. Always starting from the leftmost paddle position
2. Incrementing by 1 step to the right each iteration
3. Collecting multiple episodes at each position
4. Tracking paddle coordinates for analysis

This guarantees that all regions of the state space are explored, unlike random policy which may cluster in certain regions.

### Testing Dataset Loading

Validate the systematic dataset:

```bash
uv run python test_systematic_dataset.py
```

This will:
- Load the systematic sweep dataset
- Check data shapes and ranges
- Verify paddle position distribution
- Generate visualization: `visualizations/systematic_dataset_sample.png`

### Heatmap Visualization

The systematic sweep script automatically generates heatmaps showing paddle and ball coverage:

**Files generated:**
- `breakout_heatmaps.png` - Visual heatmap showing paddle and ball positions
- `breakout_heatmaps.npz` - Raw heatmap arrays for programmatic analysis

**What the heatmaps show:**
- **Paddle heatmap**: Horizontal coverage showing systematic left-to-right sweep
- **Ball heatmap**: Spatial coverage showing trajectory diversity

**Interpretation:**
- Uniform paddle coverage = good systematic sweep
- High ball coverage = diverse trajectories
- Gaps or clusters indicate bias or limited exploration

See [HEATMAP_TRACKING.md](../../HEATMAP_TRACKING.md) for detailed analysis guide.

### Future Improvements

1. **Adaptive Sweep Resolution**
   - Increase paddle position granularity in regions with high reward potential
   - Use fewer samples in low-value regions

2. **Action Sequence Variation**
   - After firing, vary the action policy (not just NOOP)
   - Collect trajectories with different paddle movement patterns

3. **Multi-Game Support**
   - Extend systematic sweep to other ATARI games
   - Adapt sweeping strategy to game-specific mechanics

4. **Hierarchical Coverage**
   - Start with coarse sweep (10 positions)
   - Refine with finer sweep (40 positions) after initial training

---

## Implementation Files

### Core Implementation
- `eb_jepa/datasets/atari/generate_systematic_sweep.py` - Data generation script
- `eb_jepa/datasets/atari/systematic_dataset.py` - Dataset loader class
- `eb_jepa/datasets/utils.py` - Data loading utilities with systematic dataset support

### Configuration Files
- `examples/ac_video_jepa/cfgs/train_atari.yaml` - Standard random policy training
- `examples/ac_video_jepa/cfgs/train_atari_systematic.yaml` - Systematic sweep training

### Testing and Validation
- `test_systematic_dataset.py` - Dataset loading and visualization test
- `examples/ac_video_jepa/test_planning_with_rewards.py` - Planning evaluation

---

## Troubleshooting

### Low Planning Scores

If planning scores are low:
1. Check if reward prediction loss is converging
2. Verify prediction MSE is decreasing
3. Ensure paddle position coverage is uniform (use systematic dataset)
4. Try training for more epochs (10-20)

### Memory Issues

If running out of memory:
1. Reduce `data.batch_size` (try 16 or 8)
2. Use smaller systematic dataset (`--paddle_positions 10`)
3. Disable AMP: `training.use_amp: false`

### Dataset Generation Fails

If systematic sweep generation fails:
1. Verify OCAtari is installed: `uv pip install ocatari`
2. Check output directory exists or has write permissions
3. Try reducing `--paddle_positions` for faster testing

---

## References

This systematic sweep approach is inspired by:
1. **Exploration in RL**: Systematic state space coverage improves sample efficiency
2. **Data Augmentation**: Diverse initial conditions act as implicit augmentation
3. **World Model Learning**: Better coverage → better predictions → better planning

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md)
For quick start guide, see [QUICKSTART.md](QUICKSTART.md)

**Last Updated:** March 21, 2026
