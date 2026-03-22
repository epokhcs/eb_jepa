"""
Test MPPI and CEM planning on ATARI Breakout with trained JEPA model.

This script tests planning algorithms (MPPI and CEM) for playing ATARI Breakout
using the trained world model to predict future states and select actions.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eb_jepa.architectures import (
    DiscreteActionEncoder,
    DiscreteInverseDynamicsModel,
    ImpalaEncoder,
    RNNPredictor,
)
from eb_jepa.datasets.atari.config import AtariConfig
from eb_jepa.datasets.atari.env import AtariEnv
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
import torch.nn as nn


class DiscreteMPPIPlanner:
    """MPPI planner for discrete action spaces."""

    def __init__(self, num_actions, horizon, num_samples, temperature=1.0):
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature

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
        # Sample random action sequences
        action_samples = torch.randint(
            0, self.num_actions, (self.num_samples, 1, self.horizon), device=device
        ).float()

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

                # Cost = negative "goodness" (we want high activation in latent space)
                # Higher variance in latent space = more interesting/active game state
                cost = -predicted_states.var(dim=(1, 3, 4)).mean()
                costs.append(cost.item())

        costs = np.array(costs)

        # MPPI: Weight samples by softmax of negative cost
        weights = np.exp(-costs / self.temperature)
        weights /= weights.sum()

        # Weighted mean action (approximate - use mode for discrete)
        # For discrete, we'll just return the best action sequence
        best_idx = np.argmax(weights)
        best_actions = action_samples[best_idx:best_idx+1]

        return best_actions, costs


class DiscreteCEMPlanner:
    """CEM planner for discrete action spaces."""

    def __init__(self, num_actions, horizon, num_samples, num_elites, n_iters=5):
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.n_iters = n_iters

    def plan(self, jepa, obs_init, device):
        """
        Plan action sequence using CEM with discrete actions.

        Args:
            jepa: Trained JEPA model
            obs_init: Initial observation [1, C, 1, H, W]
            device: Device

        Returns:
            best_actions: [1, 1, horizon] best action sequence
            costs_history: List of cost arrays for each iteration
        """
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

                    # Cost = negative activity in latent space
                    cost = -predicted_states.var(dim=(1, 3, 4)).mean()
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


def test_planning(checkpoint_path, planner_type="mppi", num_episodes=5, horizon=16):
    """Test planning with MPPI or CEM."""
    print(f"Testing {planner_type.upper()} planning on ATARI Breakout")
    print(f"Checkpoint: {checkpoint_path}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load checkpoint and rebuild model
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, 256, 256),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=1,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(1, 84, 84),
    )

    action_encoder = DiscreteActionEncoder(num_actions=4, embedding_dim=64)
    predictor = RNNPredictor(
        hidden_size=512,
        action_dim=64,
        num_layers=1,
        final_ln=nn.LayerNorm(512),
        action_encoder=action_encoder,
    )

    idm = DiscreteInverseDynamicsModel(state_dim=512, hidden_dim=256, num_actions=4)
    regularizer = VC_IDM_Sim_Regularizer(
        idm=idm,
        cov_coeff=8,
        std_coeff=16,
        sim_coeff_t=12,
        idm_coeff=1,
    )

    ploss = SquareLossSeq()
    jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, ploss)
    jepa.load_state_dict(checkpoint['model_state_dict'])
    jepa.to(device)
    jepa.eval()
    print("✅ Model loaded")

    # Create environment
    config = AtariConfig(game_name="Breakout", batch_size=1)
    env = AtariEnv(config)
    action_names = {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT'}

    # Create planner
    if planner_type.lower() == "mppi":
        planner = DiscreteMPPIPlanner(
            num_actions=4,
            horizon=horizon,
            num_samples=100,
            temperature=1.0
        )
    else:  # CEM
        planner = DiscreteCEMPlanner(
            num_actions=4,
            horizon=horizon,
            num_samples=100,
            num_elites=10,
            n_iters=5
        )

    print(f"✅ {planner_type.upper()} planner created (horizon={horizon})")

    # Run episodes
    results = []
    print(f"\n🎮 Running {num_episodes} episodes with {planner_type.upper()} planning...")

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        action_sequence = []

        print(f"\nEpisode {ep+1}:")
        for step in tqdm(range(200), desc=f"  Episode {ep+1}"):
            # Prepare observation for planning
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]

            # Plan action sequence
            planned_actions, costs = planner.plan(jepa, obs_tensor, device)

            # Take first action
            action = int(planned_actions[0, 0, 0].item())
            action_sequence.append(action)

            # Execute in environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if done or truncated:
                break

        results.append({
            'episode': ep + 1,
            'reward': episode_reward,
            'length': episode_length,
            'actions': action_sequence
        })

        print(f"  Reward: {episode_reward:.1f} | Length: {episode_length}")

        # Print action distribution
        action_counts = {i: action_sequence.count(i) for i in range(4)}
        action_dist = " | ".join([f"{action_names[i]}: {action_counts.get(i, 0)}" for i in range(4)])
        print(f"  Actions: {action_dist}")

    # Summary statistics
    avg_reward = np.mean([r['reward'] for r in results])
    std_reward = np.std([r['reward'] for r in results])
    avg_length = np.mean([r['length'] for r in results])

    print(f"\n{'='*60}")
    print(f"📊 {planner_type.upper()} Planning Results")
    print(f"{'='*60}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Length: {avg_length:.1f}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test planning on ATARI Breakout')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--planner', type=str, choices=['mppi', 'cem', 'both'], default='both',
                       help='Planner to test')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--horizon', type=int, default=16, help='Planning horizon')

    args = parser.parse_args()

    if args.planner == 'both':
        print("Testing both MPPI and CEM planners\n")
        mppi_results = test_planning(args.checkpoint, 'mppi', args.num_episodes, args.horizon)
        print("\n" + "="*60 + "\n")
        cem_results = test_planning(args.checkpoint, 'cem', args.num_episodes, args.horizon)

        # Compare
        mppi_avg = np.mean([r['reward'] for r in mppi_results])
        cem_avg = np.mean([r['reward'] for r in cem_results])

        print(f"\n{'='*60}")
        print(f"🏆 Comparison")
        print(f"{'='*60}")
        print(f"MPPI Average Reward: {mppi_avg:.2f}")
        print(f"CEM Average Reward:  {cem_avg:.2f}")
        print(f"Winner: {'MPPI' if mppi_avg > cem_avg else 'CEM'} (+{abs(mppi_avg - cem_avg):.2f})")
        print(f"{'='*60}")

    else:
        test_planning(args.checkpoint, args.planner, args.num_episodes, args.horizon)


if __name__ == "__main__":
    main()
