"""
Record video of planning algorithms playing Breakout.

This script runs a planner and saves a video of the best episode.
"""

import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eb_jepa.checkpoint_utils import load_jepa_from_checkpoint
from eb_jepa.datasets.atari.config import AtariConfig
from eb_jepa.datasets.atari.env import AtariEnv


def load_model_and_reward_head(checkpoint_path, device, objective):
    """Load JEPA model and optional reward head using proper checkpoint loader."""
    print("Loading checkpoint...")

    # Get config path from checkpoint directory
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.yaml"

    # Use the proper checkpoint loader (same as test_planning_with_rewards.py)
    jepa, cfg, data_config = load_jepa_from_checkpoint(
        str(checkpoint_path),
        str(config_path),
        device=device
    )

    jepa.eval()
    print(f"✅ JEPA model loaded on {device}")

    # Reward head is attached to jepa.reward_head if it exists
    reward_head = getattr(jepa, 'reward_head', None)
    if objective == "predicted_reward":
        if reward_head is None:
            raise ValueError("Reward prediction requested but no reward head found in checkpoint")
        reward_head.eval()
        print("✅ Reward head loaded")

    return jepa, reward_head


class MPPIPlanner:
    """MPPI planner for discrete actions (Atari)."""

    def __init__(self, num_actions=4, horizon=16, num_samples=100, temperature=1.0):
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature

    def plan(self, model, obs, reward_head, objective, device):
        """Plan using MPPI with correct tensor shapes."""
        # Sample actions with CORRECT shape: (num_samples, 1, horizon)
        action_samples = torch.randint(
            0, self.num_actions,
            (self.num_samples, 1, self.horizon),
            device=device
        )

        costs = []

        # Process each sample INDIVIDUALLY (not batched)
        for i in range(self.num_samples):
            actions = action_samples[i:i+1]  # Shape: (1, 1, horizon)

            with torch.no_grad():
                # Unroll with matching batch dimensions
                predicted_states = model.unroll(
                    obs,  # (1, C, 1, H, W) - batch size 1
                    actions,  # (1, 1, horizon) - batch size 1
                    nsteps=self.horizon,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=False,
                    return_all_steps=False,
                )[0]  # Shape: (1, C, horizon, H, W)

            # Compute cost based on objective
            if objective == "predicted_reward" and reward_head is not None:
                # Reward head outputs logits: (1, horizon, 2)
                logits = reward_head(predicted_states)

                # Extract probability of reward class (class 1)
                probs = torch.softmax(logits, dim=-1)[:, :, 1]  # (1, horizon)

                # Cost is negative sum (we minimize cost, so maximize reward)
                cost = -probs.sum()
            else:
                # Latent variance objective (baseline)
                cost = -predicted_states.var(dim=(1, 3, 4)).mean()

            costs.append(cost)

        # Select action with lowest cost
        costs = torch.stack(costs)
        best_idx = costs.argmin()
        best_action = action_samples[best_idx, 0, 0].item()  # Extract first action

        return best_action


def record_video(checkpoint_path, planner_type, objective, output_path, num_episodes=3, horizon=8):
    """Record video of planning."""
    print(f"\n{'='*70}")
    print(f"Recording {planner_type.upper()} with {objective.upper()} objective")
    print(f"{'='*70}\n")

    # Load model (device extraction happens here)
    jepa, reward_head = load_model_and_reward_head(checkpoint_path, 'auto', objective)

    # Extract device from loaded model
    device = next(jepa.parameters()).device
    print(f"Using device: {device}")

    # Create environment WITHOUT video wrapper (we'll manually capture frames)
    print("Creating environment...")
    config = AtariConfig(
        game_name="Breakout",
        batch_size=1,
    )
    env = AtariEnv(config, render_mode="rgb_array")

    # Create planner
    if planner_type == "mppi":
        planner = MPPIPlanner(num_actions=4, horizon=horizon, num_samples=200, temperature=0.5)
    else:
        raise NotImplementedError(f"Planner {planner_type} not implemented for video recording")

    # Run episodes
    best_reward = -float('inf')
    best_episode = None
    all_videos = []

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        frames = []

        # Capture initial frame
        rgb_frame = env.env.render()
        frames.append(rgb_frame)

        pbar = tqdm(total=200, desc=f"Episode {episode_idx+1}/{num_episodes}", leave=True)

        while not done and episode_length < 200:
            # Prepare observation
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(2).to(device)
            else:
                obs_tensor = obs.float().unsqueeze(0).unsqueeze(2).to(device)

            # Plan action
            action = planner.plan(jepa, obs_tensor, reward_head, objective, device)
            # action is already an int from the planner

            # Debug: print first few actions
            if episode_length < 5:
                print(f"  Step {episode_length}: Selected action = {action} ({['NOOP', 'FIRE', 'RIGHT', 'LEFT'][action]})")

            # Execute
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            # Capture frame after action
            rgb_frame = env.env.render()
            frames.append(rgb_frame)

            pbar.set_postfix({"reward": episode_reward, "action": ["NOOP", "FIRE", "RIGHT", "LEFT"][action]})
            pbar.update(1)

        pbar.close()

        # Save this episode's video
        video_path = output_path.parent / f"{output_path.stem}_episode{episode_idx+1}.mp4"
        print(f"Saving episode {episode_idx+1} video to {video_path}...")
        imageio.mimsave(str(video_path), frames, fps=30)
        all_videos.append(video_path)

        print(f"Episode {episode_idx+1}: Reward = {episode_reward}, Length = {episode_length}, Video frames = {len(frames)}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode_idx + 1

    env.close()

    print(f"\n{'='*70}")
    print(f"✅ Video recording complete!")
    print(f"Best episode: {best_episode} with reward {best_reward}")
    print(f"Videos saved:")
    for vp in all_videos:
        print(f"  - {vp}")
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
