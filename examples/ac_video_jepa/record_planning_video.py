"""
Record video of planning algorithms playing Breakout.

This script runs a planner and saves a video of the best episode.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eb_jepa.architectures import (
    DiscreteActionEncoder,
    DiscreteInverseDynamicsModel,
    ImpalaEncoder,
    RewardPredictionHead,
    RNNPredictor,
)
from eb_jepa.datasets.atari.config import AtariConfig
from eb_jepa.datasets.atari.env import AtariEnv
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer


def load_model_and_reward_head(checkpoint_path, device, objective):
    """Load JEPA model and optional reward head."""
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build model architecture (matching train_atari.yaml)
    action_encoder = DiscreteActionEncoder(
        num_actions=4,
        embedding_dim=64
    ).to(device)

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
    ).to(device)

    predictor = RNNPredictor(
        hidden_size=512,
        action_dim=64,
        num_layers=1,
        final_ln=nn.LayerNorm(512),
        action_encoder=action_encoder,
    ).to(device)

    # Create IDM for regularizer
    idm = DiscreteInverseDynamicsModel(state_dim=512, hidden_dim=256, num_actions=4)
    regularizer = VC_IDM_Sim_Regularizer(
        idm=idm,
        cov_coeff=8,
        std_coeff=16,
        sim_coeff_t=12,
        idm_coeff=1,
    )

    loss_fn = SquareLossSeq()
    jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, loss_fn).to(device)
    jepa.load_state_dict(checkpoint['model_state_dict'])
    jepa.eval()
    print("✅ JEPA model loaded")

    # Load reward head if using predicted_reward objective
    reward_head = None
    if objective == "predicted_reward":
        print("\nLoading reward prediction head...")
        state_dim = 512
        reward_head = RewardPredictionHead(
            state_dim=state_dim,
            hidden_dim=256,
            spatial_aggregate="mean",
        ).to(device)
        reward_head.load_state_dict(checkpoint['reward_head_state_dict'])
        reward_head.eval()
        print("✅ Reward head loaded")

    return jepa, reward_head


class MPPIPlanner:
    """Simple MPPI planner."""

    def __init__(self, num_actions=4, horizon=8, num_samples=50, temperature=1.0):
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature

    def plan(self, model, obs, reward_head, objective, device):
        """Plan using MPPI."""
        # Sample random action sequences
        actions = torch.randint(
            0, self.num_actions,
            (self.num_samples, self.horizon, 1),
            device=device
        )

        # Rollout each sequence
        obs_repeated = obs.repeat(self.num_samples, 1, 1, 1, 1)

        with torch.no_grad():
            predicted_states, _ = model.unroll(
                obs_repeated,
                actions,
                nsteps=self.horizon,
                unroll_mode="autoregressive",
                ctxt_window_time=1,
                compute_loss=False,
                return_all_steps=False,
            )

        # Compute costs
        if objective == "latent_variance":
            costs = -predicted_states.var(dim=(1, 3, 4)).mean(dim=1)
        else:  # predicted_reward
            predicted_rewards = reward_head(predicted_states)
            costs = -predicted_rewards.sum(dim=1)

        # Weight by softmax
        weights = torch.softmax(-costs / self.temperature, dim=0)

        # Return first action of best sequence
        best_idx = costs.argmin()
        return actions[best_idx, 0, 0]


def record_video(checkpoint_path, planner_type, objective, output_path, num_episodes=3, horizon=8):
    """Record video of planning."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Recording {planner_type.upper()} with {objective.upper()} objective")
    print(f"{'='*70}\n")

    # Load model
    jepa, reward_head = load_model_and_reward_head(checkpoint_path, device, objective)

    # Create environment with video recording
    print("Creating environment with video recording...")
    config = AtariConfig(
        game_name="Breakout",
        batch_size=1,
    )
    atari_wrapper = AtariEnv(config, render_mode="rgb_array")

    # Wrap the underlying gymnasium env for video recording
    from gymnasium.wrappers import RecordVideo
    atari_wrapper.env = RecordVideo(
        atari_wrapper.env,
        output_path.parent,
        name_prefix=output_path.stem,
        episode_trigger=lambda x: True  # Record all episodes
    )
    env = atari_wrapper

    # Create planner
    if planner_type == "mppi":
        planner = MPPIPlanner(num_actions=4, horizon=horizon, num_samples=50)
    else:
        raise NotImplementedError(f"Planner {planner_type} not implemented for video recording")

    # Run episodes
    best_reward = -float('inf')
    best_episode = None

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        pbar = tqdm(total=200, desc=f"Episode {episode_idx+1}/{num_episodes}", leave=True)

        while not done and episode_length < 200:
            # Prepare observation
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(2).to(device)
            else:
                obs_tensor = obs.float().unsqueeze(0).unsqueeze(2).to(device)

            # Plan action
            action = planner.plan(jepa, obs_tensor, reward_head, objective, device)
            action = int(action.item())

            # Execute
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            pbar.set_postfix({"reward": episode_reward, "action": ["NOOP", "FIRE", "RIGHT", "LEFT"][action]})
            pbar.update(1)

        pbar.close()

        print(f"Episode {episode_idx+1}: Reward = {episode_reward}, Length = {episode_length}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode_idx + 1

    env.close()

    print(f"\n{'='*70}")
    print(f"✅ Video recording complete!")
    print(f"Best episode: {best_episode} with reward {best_reward}")
    print(f"Videos saved to: {output_path.parent}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--planner', type=str, default='mppi', choices=['mppi', 'cem'])
    parser.add_argument('--objective', type=str, required=True,
                       choices=['latent_variance', 'predicted_reward'])
    parser.add_argument('--output_dir', type=str, default='videos')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Record N episodes and keep the best')
    parser.add_argument('--horizon', type=int, default=8)
    args = parser.parse_args()

    # Create output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = f"{args.planner}_{args.objective}"
    output_path = output_dir / output_name

    # Record video
    record_video(
        args.checkpoint,
        args.planner,
        args.objective,
        output_path,
        args.num_episodes,
        args.horizon
    )


if __name__ == "__main__":
    main()
