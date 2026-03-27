"""
Test MPPI and CEM planning on ATARI Breakout with REWARD PREDICTION.

This script compares two planning objectives:
1. Latent variance (baseline) - maximizes state "interestingness"
2. Predicted rewards (new) - maximizes actual game score

Usage:
    # Baseline (latent variance)
    python -m examples.ac_video_jepa.test_planning_with_rewards \
        --checkpoint path/to/checkpoint.pth.tar \
        --planner mppi \
        --objective latent_variance \
        --num_episodes 20

    # With reward prediction
    python -m examples.ac_video_jepa.test_planning_with_rewards \
        --checkpoint path/to/checkpoint.pth.tar \
        --planner mppi \
        --objective predicted_reward \
        --num_episodes 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eb_jepa.architectures import (
    DiscreteActionEncoder,
    DiscreteInverseDynamicsModel,
    ImpalaEncoder,
    RNNPredictor,
    RewardPredictionHead,
)
from eb_jepa.datasets.atari.config import AtariConfig
from eb_jepa.datasets.atari.env import AtariEnv
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
import torch.nn as nn


class DiscreteMPPIPlanner:
    """MPPI planner for discrete action spaces with configurable objectives."""

    def __init__(self, num_actions, horizon, num_samples, temperature=1.0, objective="latent_variance", reward_head=None):
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.objective = objective
        self.reward_head = reward_head

        if objective == "predicted_reward" and reward_head is None:
            raise ValueError("reward_head must be provided when objective='predicted_reward'")

    def plan(self, jepa, obs_init, device):
        """
        Plan action sequence using MPPI with discrete actions.

        Args:
            jepa: Trained JEPA model
            obs_init: Initial observation [1, C, 1, H, W]
            device: Device

        Returns:
            best_actions: [1, 1, horizon] best action sequence
            all_costs: [num_samples] costs for each sample
        """
        # Sample random action sequences (keep as long/int64 for discrete actions)
        action_samples = torch.randint(
            0, self.num_actions, (self.num_samples, 1, self.horizon), device=device
        )

        # Rollout each action sequence
        costs = []
        with torch.no_grad():
            for i in range(self.num_samples):
                actions = action_samples[i:i+1]  # [1, 1, horizon]

                # Predict future states
                predicted_states = jepa.unroll(
                    obs_init,
                    actions,
                    nsteps=self.horizon,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=False,
                    return_all_steps=False,
                )[0]  # [1, D, horizon, H', W']

                # Compute cost based on objective
                if self.objective == "latent_variance":
                    # Baseline: maximize latent variance (proxy for interesting states)
                    cost = -predicted_states.var(dim=(1, 3, 4)).mean()
                elif self.objective == "predicted_reward":
                    # New: maximize predicted cumulative reward
                    # Reward head outputs logits [1, horizon, 2] for binary classification
                    logits = self.reward_head(predicted_states)  # [1, horizon, 2]
                    # Get probability of reward class (index 1)
                    probs = torch.softmax(logits, dim=-1)[:, :, 1]  # [1, horizon]
                    # Maximize expected cumulative reward
                    cost = -probs.sum()  # Negative because we minimize cost
                else:
                    raise ValueError(f"Unknown objective: {self.objective}")

                costs.append(cost.item())

        costs = np.array(costs)

        # MPPI: Weight samples by softmax of negative cost
        weights = np.exp(-costs / self.temperature)
        weights /= weights.sum()

        # For discrete, we'll just return the best action sequence
        best_idx = np.argmax(weights)
        best_actions = action_samples[best_idx:best_idx+1]

        return best_actions, costs


class DiscreteCEMPlanner:
    """CEM planner for discrete action spaces with configurable objectives."""

    def __init__(self, num_actions, horizon, num_samples, num_elites, n_iters=5, objective="latent_variance", reward_head=None):
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.n_iters = n_iters
        self.objective = objective
        self.reward_head = reward_head

        if objective == "predicted_reward" and reward_head is None:
            raise ValueError("reward_head must be provided when objective='predicted_reward'")

    def plan(self, jepa, obs_init, device):
        """Plan action sequence using CEM."""
        # Initialize uniform distribution over actions
        action_probs = torch.ones(
            self.horizon, self.num_actions, device=device
        ) / self.num_actions

        costs_history = []

        for iteration in range(self.n_iters):
            # Sample action sequences from current distribution
            action_samples = []
            for _ in range(self.num_samples):
                actions = []
                for t in range(self.horizon):
                    action = torch.multinomial(action_probs[t], 1).item()
                    actions.append(action)
                action_samples.append(actions)

            action_samples = torch.tensor(
                action_samples, dtype=torch.float32, device=device
            ).unsqueeze(1)  # [num_samples, 1, horizon]

            # Evaluate each action sequence
            costs = []
            with torch.no_grad():
                for i in range(self.num_samples):
                    actions = action_samples[i:i+1]  # [1, 1, horizon]

                    # Predict future states
                    predicted_states = jepa.unroll(
                        obs_init,
                        actions,
                        nsteps=self.horizon,
                        unroll_mode="autoregressive",
                        ctxt_window_time=1,
                        compute_loss=False,
                        return_all_steps=False,
                    )[0]  # [1, D, horizon, H', W']

                    # Compute cost based on objective
                    if self.objective == "latent_variance":
                        # Baseline: maximize latent variance
                        cost = -predicted_states.var(dim=(1, 3, 4)).mean()
                    elif self.objective == "predicted_reward":
                        # New: maximize predicted cumulative reward
                        # Reward head outputs logits [1, horizon, 2] for binary classification
                        logits = self.reward_head(predicted_states)  # [1, horizon, 2]
                        # Get probability of reward class (index 1)
                        probs = torch.softmax(logits, dim=-1)[:, :, 1]  # [1, horizon]
                        # Maximize expected cumulative reward
                        cost = -probs.sum()  # Negative because we minimize cost
                    else:
                        raise ValueError(f"Unknown objective: {self.objective}")

                    costs.append(cost.item())

            costs = np.array(costs)
            costs_history.append(costs)

            # Select elites
            elite_indices = np.argsort(costs)[:self.num_elites]
            elite_actions = action_samples[elite_indices].long()  # [num_elites, 1, horizon]

            # Update action distribution based on elites
            new_probs = torch.zeros_like(action_probs)
            for t in range(self.horizon):
                elite_actions_t = elite_actions[:, 0, t]
                for action_idx in range(self.num_actions):
                    count = (elite_actions_t == action_idx).sum().item()
                    new_probs[t, action_idx] = count

            # Normalize and add smoothing
            new_probs = new_probs + 0.1
            new_probs = new_probs / new_probs.sum(dim=1, keepdim=True)
            action_probs = new_probs

        # Return best elite action
        best_idx = elite_indices[0]
        best_actions = action_samples[best_idx:best_idx+1]

        return best_actions, costs_history


def test_planning(checkpoint_path, planner_type="mppi", objective="latent_variance", num_episodes=5, horizon=16):
    """Test planning with MPPI or CEM."""
    print(f"\n{'='*70}")
    print(f"Testing {planner_type.upper()} planning with {objective.upper()} objective")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}, Horizon: {horizon}")

    # Find config.yaml in the same directory as checkpoint
    from pathlib import Path
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}. Expected to find config.yaml in checkpoint directory.")

    # Load JEPA model using checkpoint_utils
    print("\nLoading JEPA model from checkpoint...")
    from eb_jepa.checkpoint_utils import load_jepa_from_checkpoint

    jepa, cfg, data_config = load_jepa_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=str(config_path),
        device='auto'
    )

    device = next(jepa.parameters()).device
    print(f"Device: {device}")
    print("✅ JEPA model loaded")

    # Get reward head from JEPA (already loaded by load_jepa_from_checkpoint)
    reward_head = None
    if objective == "predicted_reward":
        print("\nChecking reward prediction head...")
        if hasattr(jepa, 'reward_head') and jepa.reward_head is not None:
            reward_head = jepa.reward_head
            print("✅ Reward head loaded from checkpoint")
        else:
            print("⚠️  WARNING: No reward head in checkpoint!")
            print("    Using random initialization - results will be poor.")
            print("    Please train with reward_prediction=true first.")

            # Create random reward head as fallback
            state_dim = 512  # From encoder mlp_output_dim
            reward_head = RewardPredictionHead(
                state_dim=state_dim,
                hidden_dim=256,
                spatial_aggregate="mean",
            ).to(device)

        reward_head.eval()

    # Create environment
    print("\nInitializing environment...")
    env_cfg = AtariConfig(
        game_name="Breakout",
        img_size=84,
        grayscale=True,
        frame_skip=4,
        normalize=True,
    )
    env = AtariEnv(config=env_cfg)

    # Create planner
    print(f"\nCreating {planner_type.upper()} planner...")
    if planner_type == "mppi":
        planner = DiscreteMPPIPlanner(
            num_actions=4,
            horizon=horizon,
            num_samples=100,
            temperature=1.0,
            objective=objective,
            reward_head=reward_head,
        )
    elif planner_type == "cem":
        planner = DiscreteCEMPlanner(
            num_actions=4,
            horizon=horizon,
            num_samples=100,
            num_elites=10,
            n_iters=5,
            objective=objective,
            reward_head=reward_head,
        )
    else:
        raise ValueError(f"Unknown planner: {planner_type}")

    # Run episodes
    print(f"\n{'='*70}")
    print(f"RUNNING {num_episodes} EPISODES")
    print(f"{'='*70}\n")

    episode_rewards = []
    episode_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # NOOP, FIRE, RIGHT, LEFT

    for episode_idx in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        pbar = tqdm(total=200, desc=f"Episode {episode_idx+1}/{num_episodes}", leave=True)

        while not done and episode_length < 200:
            # Prepare observation
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(2).to(device)  # [1, C, 1, H, W]
            else:
                # Already a tensor
                obs_tensor = obs.float().unsqueeze(0).unsqueeze(2).to(device)  # [1, C, 1, H, W]

            # Plan action
            planned_actions, costs = planner.plan(jepa, obs_tensor, device)

            # Execute first action
            action = int(planned_actions[0, 0, 0].item())
            action_counts[action] += 1

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'action': ['NOOP', 'FIRE', 'RIGHT', 'LEFT'][action]
            })

        pbar.close()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"  Episode {episode_idx+1}: Reward = {episode_reward:.1f}, Length = {episode_length}")

    # Print statistics
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nObjective: {objective}")
    print(f"Planner: {planner_type.upper()}")
    print(f"\n📊 Episode Rewards:")
    print(f"  Mean:   {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Median: {np.median(episode_rewards):.2f}")
    print(f"  Min:    {np.min(episode_rewards):.2f}")
    print(f"  Max:    {np.max(episode_rewards):.2f}")

    print(f"\n📈 Episode Lengths:")
    print(f"  Mean:   {np.mean(episode_lengths):.1f}")

    print(f"\n🎮 Action Distribution:")
    total_actions = sum(action_counts.values())
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    for action_idx, action_name in enumerate(action_names):
        count = action_counts[action_idx]
        pct = 100.0 * count / total_actions if total_actions > 0 else 0
        print(f"  {action_name:>6s}: {count:>5d} ({pct:>5.1f}%)")

    # Save results
    results_dir = Path(checkpoint_path).parent / "planning_results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"{planner_type}_{objective}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Objective: {objective}\n")
        f.write(f"Planner: {planner_type}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Mean reward: {np.mean(episode_rewards):.2f}\n")
        f.write(f"Std reward: {np.std(episode_rewards):.2f}\n")
        f.write(f"Action distribution:\n")
        for action_idx, action_name in enumerate(action_names):
            count = action_counts[action_idx]
            pct = 100.0 * count / total_actions if total_actions > 0 else 0
            f.write(f"  {action_name}: {pct:.1f}%\n")

    print(f"\n✅ Results saved to {results_file}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Episode rewards
    ax = axes[0]
    ax.plot(range(1, num_episodes+1), episode_rewards, marker='o')
    ax.axhline(np.mean(episode_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'{planner_type.upper()} with {objective}\nEpisode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Action distribution
    ax = axes[1]
    ax.bar(action_names, [action_counts[i] / total_actions * 100 for i in range(4)])
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Action Distribution')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_file = results_dir / f"{planner_type}_{objective}_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to {plot_file}")

    plt.show()

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'action_counts': action_counts,
    }


def main():
    parser = argparse.ArgumentParser(description='Test planning with reward prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--planner', type=str, default='mppi', choices=['mppi', 'cem'],
                       help='Planner to use (mppi or cem)')
    parser.add_argument('--objective', type=str, default='latent_variance',
                       choices=['latent_variance', 'predicted_reward'],
                       help='Planning objective (latent_variance or predicted_reward)')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--horizon', type=int, default=16,
                       help='Planning horizon')

    args = parser.parse_args()

    test_planning(
        checkpoint_path=args.checkpoint,
        planner_type=args.planner,
        objective=args.objective,
        num_episodes=args.num_episodes,
        horizon=args.horizon,
    )


if __name__ == '__main__':
    main()
