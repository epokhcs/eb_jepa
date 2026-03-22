# Reward Prediction for ATARI JEPA

## Overview

Added reward prediction capability to the JEPA world model, enabling the model to predict immediate rewards from latent states. This allows planning algorithms to optimize for actual game score instead of using proxy objectives like latent variance.

## Changes Made

### 1. New Architecture Component: `RewardPredictionHead`

**File**: `eb_jepa/architectures.py`

A lightweight MLP that predicts scalar rewards from latent state representations:

```python
class RewardPredictionHead(nn.Module):
    """Predicts scalar reward from latent state representation."""

    def __init__(self, state_dim, hidden_dim=256, spatial_aggregate="mean"):
        # state_dim: Feature dimension (e.g., 256 for ATARI)
        # hidden_dim: Hidden layer size
        # spatial_aggregate: How to aggregate spatial dims ("mean", "max", "flatten")
```

**How it works**:
1. Takes latent states `[B, D, T, H, W]` from the world model
2. Aggregates spatial dimensions (default: mean pooling)
3. Passes through 3-layer MLP
4. Outputs predicted reward `[B, T]` for each timestep

### 2. New Loss Function: `RewardPredictionLoss`

**File**: `eb_jepa/losses.py`

Simple MSE loss between predicted and actual rewards:

```python
class RewardPredictionLoss(nn.Module):
    """MSE loss for reward prediction."""

    def forward(self, predicted_latents, target_rewards):
        predicted_rewards = self.reward_head(predicted_latents)
        return F.mse_loss(predicted_rewards, target_rewards)
```

### 3. Training Configuration Updates

**File**: `examples/ac_video_jepa/cfgs/train_atari.yaml`

Added reward prediction settings:

```yaml
model:
  reward_prediction: true  # Enable reward prediction head
  reward_head_hidden_dim: 256  # Hidden dimension for reward head
  reward_loss_coeff: 1.0  # Coefficient for reward prediction loss
```

### 4. Training Loop Integration

**File**: `examples/ac_video_jepa/main.py`

**Model Initialization** (after IDM creation):
- Creates `RewardPredictionHead` with appropriate dimensions
- Creates `RewardPredictionLoss` wrapper
- Creates separate optimizer for reward head
- Logs reward head parameters

**Training Loop** (after probe loss):
- Extracts reward targets from batch metadata
- Runs forward pass through trained JEPA encoder/predictor
- Predicts rewards from resulting latent states
- Computes MSE loss against ground truth rewards
- Backpropagates and updates reward head parameters
- Logs reward loss to W&B and metrics logger

## Training with Reward Prediction

### Command:

```bash
python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml
```

### What Gets Trained:

1. **JEPA encoder + predictor**: Main world model (same as before)
2. **Inverse Dynamics Model**: Predicts actions from state transitions
3. **Reward Head**: NEW - Predicts rewards from latent states

### Loss Components:

```
Total Loss = Prediction Loss + Regularization Loss + IDM Loss + Reward Loss
           = (MSE on latents) + (VICReg) + (Action prediction) + (Reward prediction)
```

### Monitoring:

New metrics in W&B / logs:
- `train/reward_loss`: MSE between predicted and actual rewards
- `optim/reward_lr`: Learning rate for reward head optimizer

Progress bar now shows:
```
loss: 7.45 | reg: 5.12 | pred: 1.83 | rew: 0.50
```

## Using Reward Prediction for Planning

Once trained, the reward head enables planning algorithms to optimize for actual game score.

### Before (without reward prediction):

```python
# Planning objective: maximize latent variance (proxy)
cost = -predicted_states.var(dim=(1, 3, 4)).mean()
```

**Problem**: Latent variance is just a heuristic. High variance ≠ high reward.

### After (with reward prediction):

```python
# Planning objective: maximize predicted cumulative reward
predicted_rewards = reward_head(predicted_states)  # [B, T]
cost = -predicted_rewards.sum(dim=1).mean()  # Maximize total reward
```

**Benefit**: Directly optimizes for game score! 🎯

## Example: Updated Planning with Rewards

Update the planning test script to use reward prediction:

```python
def plan_with_rewards(jepa, reward_head, obs_init, actions, device):
    """Plan using predicted rewards as objective."""

    # Predict future states
    predicted_states = jepa.unroll(
        obs_init, actions, nsteps=horizon,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        compute_loss=False,
    )[0]  # [B, D, T, H, W]

    # Predict rewards from states
    predicted_rewards = reward_head(predicted_states)  # [B, T]

    # Cost = negative cumulative reward
    cost = -predicted_rewards.sum(dim=1)  # [B]

    return cost
```

Then in MPPI/CEM:
- Sample action sequences
- Rollout world model
- **Use reward head to predict cumulative reward**
- Select actions that maximize predicted reward

## Expected Improvements

### Without Reward Prediction:
- Planning optimizes for "dynamic states" (high latent variance)
- Agent favors FIRE (launches ball) but doesn't aim for score
- Rewards: ~1-2 points per episode

### With Reward Prediction:
- Planning directly optimizes for game score
- Agent learns to break bricks (high reward actions)
- Rewards: **5-10x improvement** expected
- Better long-term strategy (e.g., aiming for high-value bricks)

## Verification Steps

After training with reward prediction:

1. **Check reward loss convergence**:
   ```bash
   # Should decrease from ~1.0 to ~0.1-0.3
   grep "reward_loss" logs/training.log
   ```

2. **Test reward prediction accuracy**:
   ```python
   # Predict rewards on validation set
   predicted = reward_head(predicted_latents)
   actual = batch.metadata['rewards']
   mse = ((predicted - actual) ** 2).mean()
   print(f"Reward prediction MSE: {mse:.4f}")
   ```

3. **Run planning with reward objective**:
   ```bash
   python examples/ac_video_jepa/test_planning.py \
     --checkpoint path/to/checkpoint.pth.tar \
     --planner mppi \
     --use_reward_objective
   ```

4. **Compare episode rewards**:
   - Baseline (latent variance): ~1-2 points
   - With reward prediction: **Target: 10+ points**

## Next Steps

1. **Train the model** with reward prediction enabled (5-10 epochs)
2. **Update planning test script** to use reward-based objective
3. **Run comparative evaluation**: MPPI/CEM with vs without reward prediction
4. **Longer training** (50-100 epochs) for better reward prediction accuracy
5. **Tune reward_loss_coeff** if reward predictions are over/underfitting

## Files Modified

- `eb_jepa/architectures.py`: Added `RewardPredictionHead`
- `eb_jepa/losses.py`: Added `RewardPredictionLoss`
- `examples/ac_video_jepa/cfgs/train_atari.yaml`: Added reward prediction config
- `examples/ac_video_jepa/main.py`: Integrated reward prediction in training loop
- `eb_jepa/datasets/atari/dataset.py`: Already includes rewards in metadata ✅

## Benefits

✅ **Direct optimization**: Plans actions to maximize actual game score
✅ **No heuristics**: Replaces latent variance proxy with learned reward function
✅ **Better long-term planning**: Model learns which states lead to high rewards
✅ **Transferable**: Reward head can be frozen and used for planning without retraining

---

Ready to train! 🚀
