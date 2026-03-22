# System Architecture

Complete guide to the EB-JEPA ATARI Breakout architecture, including all components, design choices, and data flow.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Design Choices Explained](#design-choices-explained)
5. [Parameter Breakdown](#parameter-breakdown)
6. [Loss Functions](#loss-functions)

---

## Architecture Overview

### High-Level System

The ATARI Breakout implementation uses a **Joint Embedding Predictive Architecture (JEPA)** world model that learns to:

1. **Encode observations** into latent representations
2. **Predict future latents** from current state + actions
3. **Predict rewards** to enable goal-directed planning
4. **Plan actions** using population-based optimization

```
┌───────────────────────────────────────────────────────────┐
│               COMPLETE JEPA PIPELINE                      │
└───────────────────────────────────────────────────────────┘

ATARI Breakout Environment
         ↓
Grayscale Frames [84×84] + Discrete Actions [4 choices]
         ↓
┌───────────────────────────────────────────────────────────┐
│ ENCODER (IMPALA CNN) - 22M params                         │
│   Input:  [B, 1, 84, 84]                                  │
│   Output: [B, 512, 1, 1] latent per frame                 │
└───────────────────────────────────────────────────────────┘
         ↓
┌───────────────────────────────────────────────────────────┐
│ PREDICTOR (RNN with GRU) - 889K params                    │
│   Input:  Latent + Action Embedding [64D]                 │
│   Output: [B, 512, T, 1, 1] predicted future latents      │
└───────────────────────────────────────────────────────────┘
         ↓                              ↓
┌────────────────────┐     ┌────────────────────────────────┐
│ REGULARIZER        │     │ REWARD HEAD - 200K params      │
│ - Variance Loss    │     │   Aggregation + 3-layer MLP    │
│ - Covariance Loss  │     │   Output: [B, T] rewards       │
│ - Temporal Similar.│     └────────────────────────────────┘
│ - IDM (actions)    │                  ↓
└────────────────────┘     ┌────────────────────────────────┐
                           │ PLANNING (MPPI/CEM)            │
                           │   Optimize actions for reward  │
                           └────────────────────────────────┘
```

**Total:** 23.1M parameters (~260MB checkpoint)

---

## Core Components

### 1. Encoder: ImpalaEncoder

**Purpose:** Converts raw 84×84 grayscale frames into compact 512-dimensional latent representations.

**Architecture:**
```
Input: [B, 1, 84, 84] grayscale frames

Stack 1:
  Conv2d(1 → 16, kernel=8×8, stride=4)      → [B, 16, 21, 21]
  ResBlock(16) × 2
  MaxPool2d(kernel=3×3, stride=2)            → [B, 16, 10, 10]

Stack 2:
  Conv2d(16 → 256, kernel=4×4, stride=2)    → [B, 256, 5, 5]
  ResBlock(256) × 2
  MaxPool2d(kernel=3×3, stride=2)            → [B, 256, 2, 2]

Stack 3:
  Conv2d(256 → 256, kernel=3×3, stride=1)   → [B, 256, 2, 2]
  ResBlock(256) × 2
  MaxPool2d(kernel=3×3, stride=2)            → [B, 256, 1, 1]

MLP:
  AdaptiveAvgPool2d(1×1)                     → [B, 256, 1, 1]
  Flatten()                                   → [B, 256]
  Linear(256 → 512)                          → [B, 512]
  ReLU + LayerNorm

Output: [B, 512, 1, 1, 1] latent states
```

**Parameters:** 21,218,528 (92.2% of total model)

**Design Notes:**
- Progressive downsampling: 84 → 42 → 21 → 11 → 1
- ResBlocks provide skip connections for gradient flow
- Final spatial dimension of 1×1 forces spatial summarization
- 512D latent balances expressiveness and efficiency

---

### 2. Action Encoder: DiscreteActionEncoder

**Purpose:** Converts discrete action indices to continuous 64-dimensional embeddings.

**Implementation:**
```python
nn.Embedding(num_actions=4, embedding_dim=64)
```

**Action Mapping (Breakout):**
- **0: NOOP** - Do nothing, wait
- **1: FIRE** - Launch ball (critical for gameplay)
- **2: RIGHT** - Move paddle right
- **3: LEFT** - Move paddle left

**Why Embeddings?**
- Discrete actions → continuous space for planning
- Learnable representations (not one-hot)
- Enables smooth interpolation in latent space
- Joint optimization with world model

**Parameters:** 256 (4 actions × 64 dims)

---

### 3. Predictor: RNNPredictor

**Purpose:** Predicts future latent states given current state and action sequence.

**Architecture:**
```
Input:
  - Initial latent: [B, 512, 1, 1]
  - Action sequence: [B, 1, T] (discrete indices)

Action Embedding:
  Embedding(4 → 64)                          → [B, 64, T]

GRU Cell (Autoregressive):
  For each timestep t:
    action_embed_t = action_encoder(action_t) → [B, 64]
    h_t = GRU(action_embed_t, h_{t-1})       → [B, 512]

LayerNorm(512):
  Stabilize outputs

Output: [B, 512, T, 1, 1] predicted latent sequence
```

**Parameters:** 889,088

**Key Features:**
- **Single-step prediction:** Prevents error accumulation
- **Autoregressive:** Feeds predictions back for multi-step rollouts
- **GRU hidden state:** Maintains temporal memory
- **Action-conditioned:** Integrates action embeddings at each step

---

### 4. Inverse Dynamics Model (IDM)

**Purpose:** Predicts action taken between two consecutive states (auxiliary task).

**Architecture:**
```
Input: State transition (s_t, s_{t+1})
  - Concatenate latents: [B, 512+512] → [B, 1024]

MLP:
  Linear(1024 → 256) + ReLU
  Linear(256 → 256) + ReLU
  Linear(256 → 4)           → [B, 4] action logits

Loss: CrossEntropyLoss(logits, ground_truth_action)
```

**Purpose:**
- Ensures latents encode action-relevant information
- Auxiliary task improves representation quality
- Discrete actions → classification loss

---

### 5. Reward Prediction Head

**Purpose:** Predicts immediate reward from latent state to enable goal-directed planning.

**Architecture:**
```
Input: Predicted latents [B, 512, T, H, W]

Spatial Aggregation:
  Mean pooling over (H, W)                   → [B, 512, T]

MLP (per timestep):
  Linear(512 → 256) + ReLU
  Linear(256 → 128) + ReLU
  Linear(128 → 1)                            → [B, T, 1]

Output: Predicted rewards [B, T]
```

**Parameters:** ~200K (< 1% of total)

**Training:**
- Supervised with ground truth rewards from environment
- MSE loss: `(predicted_reward - actual_reward)^2`
- Separate optimizer (AdamW, same lr as main model)

**Usage:**
- During planning: `cost = -sum(reward_head(predicted_latents))`
- Directly optimizes for game score!

---

### 6. Planning Algorithms

#### MPPI (Model Predictive Path Integral)

**Algorithm:** One-shot sampling with weighted action selection

**Process:**
```
1. Sample N action sequences (categorical, N=100)
2. Rollout world model for each sequence
3. Evaluate cost (negative predicted reward)
4. Weight samples by softmax(cost / temperature)
5. Select action with best cost
```

**Speed:** ~2.2 steps/second
**Best for:** Reward-based planning, real-time applications

#### CEM (Cross-Entropy Method)

**Algorithm:** Iterative optimization of action distribution

**Process:**
```
1. Initialize uniform action logits
2. For K iterations (K=10):
   a. Sample N sequences from categorical distribution
   b. Evaluate costs via world model
   c. Keep top-E elite sequences (E=10)
   d. Update logits based on elite action frequencies
3. Return mode (argmax) of final distribution
```

**Speed:** ~0.17 steps/second (12x slower than MPPI)
**Best for:** Latent variance objective, quality > speed

**Note:** CEM fails with reward prediction (converges to NOOP), use MPPI instead!

---

## Data Flow

### Training Phase

```
1. Environment Interaction:
   ATARI Env → Trajectories [observations, actions, rewards]
   Random policy, 17-frame sequences

2. Encoding:
   Observations [B, 1, T, 84, 84] → Encoder → Latents [B, 512, T, 1, 1]

3. Prediction:
   Latents_t0 + Actions [B, 1, T] → Predictor → Predicted_Latents [B, 512, T, 1, 1]

4. Loss Computation:
   a. Prediction Loss: MSE(Predicted_Latents, Actual_Latents)
   b. Regularization: Variance + Covariance + Temporal + IDM
   c. Reward Loss: MSE(Predicted_Rewards, Actual_Rewards)

5. Optimization:
   Total_Loss = Pred + Reg + Reward
   Backprop → Update encoder, predictor, reward_head

6. Checkpoint:
   Save every epoch: encoder, predictor, IDM, reward_head, optimizers
```

### Planning Phase

```
1. Load Checkpoint:
   Restore encoder, predictor, reward_head weights

2. Get Current Observation:
   ATARI Env → Frame [1, 84, 84]

3. Encode:
   Frame → Encoder → Current_Latent [512, 1, 1]

4. Sample Action Sequences:
   MPPI: N=100 random sequences, horizon=8
   CEM: Iteratively sample and refine

5. Rollout World Model:
   For each action sequence:
     Latent_t0 + Actions → Predictor → Predicted_Latents [512, 8, 1, 1]

6. Evaluate Cost:
   Latent_variance: cost = -predicted_latents.var()
   Predicted_reward: cost = -reward_head(predicted_latents).sum()

7. Select Best Action:
   action = argmin(cost) for first timestep

8. Execute & Repeat:
   Execute action → New observation → Repeat from step 3
```

---

## Design Choices Explained

### Why JEPA over VAE/Autoencoder?

**JEPA learns from latent targets, not pixel reconstruction.**

**Advantages:**
- ✅ **No decoder needed:** Saves 50% of parameters
- ✅ **Better representations:** Learns predictive features
- ✅ **Avoids blurry predictions:** Latent space is cleaner
- ✅ **Faster training:** No pixel-level reconstruction

**Trade-off:**
- ❌ **Cannot visualize predictions directly:** Need decoder for that

**Why it matters:** Pixel reconstruction is expensive and often unnecessary for control tasks.

---

### Why IMPALA Encoder?

**Proven architecture for ATARI with efficient design.**

**Advantages:**
- ✅ **Proven:** Used in DeepMind's IMPALA agent
- ✅ **Efficient:** 22M params for 84×84 images is reasonable
- ✅ **Good spatial features:** ResNet blocks maintain structure
- ✅ **Progressive downsampling:** Captures multi-scale features

**Alternative:** Could use ResNet-18, but IMPALA is optimized for games.

---

### Why RNN Predictor (not Transformer)?

**Single-step autoregressive with GRU prevents error accumulation.**

**Advantages:**
- ✅ **Stable:** Errors don't compound as much as parallel prediction
- ✅ **Memory:** GRU hidden state acts as temporal memory
- ✅ **Simple:** Fewer hyperparameters than Transformer
- ✅ **Efficient:** Lower computational cost

**Trade-off:**
- ❌ **Slower:** Sequential, can't parallelize like Transformer
- ❌ **Limited context:** Only uses hidden state, not full history

**Why it matters:** For short horizons (8-16 steps), RNN is more stable.

---

### Why Discrete Action Embeddings?

**ATARI has discrete action space (4 actions).**

**Why embeddings > one-hot:**
- ✅ **Learnable:** Model learns action representations jointly
- ✅ **Dense:** 64D embeddings vs 4D sparse one-hot
- ✅ **Smooth planning:** Enables soft action distributions
- ✅ **Better generalization:** Similar actions have similar embeddings

**Example:** LEFT and RIGHT might learn similar embeddings (both paddle movements).

---

### Why Reward Prediction Head?

**Enables goal-directed planning for game optimization.**

**Without reward head:**
- Use heuristic objectives (e.g., maximize latent variance)
- Explores but doesn't optimize for score

**With reward head:**
- ✅ **Direct optimization:** Maximize predicted game score
- ✅ **Learned objective:** No hand-crafted heuristics
- ✅ **Lightweight:** Only 200K params (< 1% overhead)
- ✅ **Joint training:** No separate training phase

**Result:** +20% game score improvement validated!

---

### Why MPPI/CEM for Planning?

**Population-based planners handle non-differentiable objectives.**

**Why not gradient-based (like backprop through model)?**
- ❌ Discrete actions are non-differentiable
- ❌ Environment is non-differentiable
- ❌ Stochastic dynamics complicate gradients

**MPPI/CEM advantages:**
- ✅ **Model-based:** Use world model to evaluate actions
- ✅ **Sample-efficient:** 100 samples per planning step
- ✅ **No gradients:** Works with discrete actions
- ✅ **Robust:** Handles stochasticity

**MPPI vs CEM:**
- **MPPI:** Faster, one-shot, better with reward prediction
- **CEM:** Slower, iterative, better with latent variance

---

### Why Grayscale Input?

**Standard ATARI preprocessing reduces computation.**

**Advantages:**
- ✅ **Efficiency:** 1 channel vs 3 RGB channels
- ✅ **Standard:** Matches DQN/IMPALA preprocessing
- ✅ **Sufficient:** Color not needed for Breakout

**Impact:** 3x fewer encoder parameters for first layer.

---

### Why Mixed Precision (bfloat16)?

**Training efficiency without loss of quality.**

**Advantages:**
- ✅ **2-4x speedup:** Faster matrix operations on GPU
- ✅ **Less memory:** Can use larger batch sizes
- ✅ **Stable:** bfloat16 has better range than float16

**Note:** MPS (Apple Silicon) may not support all mixed precision ops.

---

## Parameter Breakdown

### Complete Model Statistics

| Component | Parameters | Percentage | Checkpoint Size |
|-----------|-----------|------------|-----------------|
| **Encoder (IMPALA)** | 21,218,528 | 92.2% | ~84 MB |
| **Predictor (RNN)** | 889,088 | 3.9% | ~3.5 MB |
| **Action Embeddings** | 256 | <0.01% | ~1 KB |
| **IDM** | 525,316 | 2.3% | ~2 MB |
| **Reward Head** | 197,505 | 0.9% | ~0.8 MB |
| **Total** | **23,010,693** | 100% | **~260 MB** |

**Checkpoint includes:**
- Model weights (260 MB)
- Optimizer states (AdamW)
- Scheduler state
- Training metadata

---

## Loss Functions

### 1. Prediction Loss (SquareLossSeq)

**Purpose:** Ensure predicted latents match actual latents.

**Formula:**
```
L_pred = MSE(predicted_latents, actual_latents)
       = mean((predicted - actual)^2)
```

**Weight:** 1.0 (unweighted, primary loss)

---

### 2. Variance Loss (VCLoss)

**Purpose:** Prevent representation collapse (all features identical).

**Formula:**
```
L_var = mean(max(0, 1 - std(features, dim=batch)))
```

**Weight:** std_coeff = 16

**Effect:** Encourages diverse features across batch.

---

### 3. Covariance Loss (VCLoss)

**Purpose:** Decorrelate features (avoid redundancy).

**Formula:**
```
Cov = (features - mean).T @ (features - mean) / (batch_size - 1)
L_cov = sum(Cov^2) - sum(diag(Cov)^2)  # Off-diagonal elements
```

**Weight:** cov_coeff = 8

**Effect:** Features learn different aspects of state.

---

### 4. Temporal Similarity Loss

**Purpose:** Encourage smooth representations over time.

**Formula:**
```
L_sim = -cosine_similarity(latent_t, latent_{t+1})
```

**Weight:** sim_coeff_t = 12

**Effect:** Prevents drastic changes between consecutive frames.

---

### 5. Inverse Dynamics Model Loss

**Purpose:** Predict actions from state transitions.

**Formula:**
```
L_idm = CrossEntropy(predicted_action_logits, actual_action)
```

**Weight:** idm_coeff = 1

**Effect:** Latents must encode action-relevant information.

---

### 6. Reward Prediction Loss

**Purpose:** Predict game rewards from latents.

**Formula:**
```
L_reward = MSE(predicted_rewards, actual_rewards)
```

**Weight:** reward_loss_coeff = 1.0

**Effect:** Enables goal-directed planning.

---

### Total Loss

```python
total_loss = (
    prediction_loss * 1.0 +
    regularizer_loss +  # (variance + covariance + temporal + idm)
    reward_loss * 1.0
)
```

**Typical values after training:**
- `prediction_loss`: ~0.25
- `regularizer_loss`: ~3.0
- `reward_loss`: ~0.0055
- `total_loss`: ~3.25

---

## Visual Summary

See [diagrams/](diagrams/) for:
- **workflow.txt** - Complete training workflow ASCII diagram
- **architecture.txt** - Detailed component architecture diagram

---

## Next Steps

- **[TRAINING.md](TRAINING.md)** - Configuration and training details
- **[EVALUATION.md](EVALUATION.md)** - Planning algorithms and evaluation
- **[README.md](README.md)** - Return to main documentation

---

**Last Updated:** March 21, 2026
