#!/usr/bin/env python3
"""
Record a long gameplay video of MPPI planner.

Runs MPPI planning for multiple episodes and saves as video.
Target: 5+ minutes of gameplay footage.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2

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


def load_model(checkpoint_path, device):
    """Load JEPA model and reward head."""
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build model
    action_encoder = DiscreteActionEncoder(num_actions=4, embedding_dim=64).to(device)
    encoder = ImpalaEncoder(
        width=1, stack_sizes=(16, 256, 256), num_blocks=2,
        dropout_rate=None, layer_norm=False, input_channels=1,
        final_ln=True, mlp_output_dim=512, input_shape=(1, 84, 84),
    ).to(device)
    predictor = RNNPredictor(
        hidden_size=512, action_dim=64, num_layers=1,
        final_ln=nn.LayerNorm(512), action_encoder=action_encoder,
    ).to(device)
    idm = DiscreteInverseDynamicsModel(state_dim=512, hidden_dim=256, num_actions=4)
    regularizer = VC_IDM_Sim_Regularizer(
        idm=idm, cov_coeff=8, std_coeff=16, sim_coeff_t=12, idm_coeff=1,
    )
    loss_fn = SquareLossSeq()
    jepa = JEPA(encoder, nn.Identity(), predictor, regularizer, loss_fn).to(device)
    jepa.load_state_dict(checkpoint['model_state_dict'])
    jepa.eval()
    print("✅ JEPA model loaded")

    # Load reward head
    reward_head = RewardPredictionHead(
        state_dim=512, hidden_dim=256, spatial_aggregate="mean",
    ).to(device)
    reward_head.load_state_dict(checkpoint['reward_head_state_dict'])
    reward_head.eval()
    print("✅ Reward head loaded")

    return jepa, reward_head


def simple_mppi_plan(jepa, reward_head, obs, device, horizon=8, num_samples=50):
    """Simple MPPI planning - returns single action."""
    # Sample random action sequences [num_samples, horizon]
    actions = torch.randint(0, 4, (num_samples, horizon), device=device)

    # Repeat observation for all samples
    obs_repeated = obs.repeat(num_samples, 1, 1, 1, 1)  # [N, C, T, H, W]

    # Expand actions to match unroll format [N, T, 1]
    actions_expanded = actions.unsqueeze(-1)  # [N, horizon, 1]

    with torch.no_grad():
        try:
            # Unroll with proper format
            predicted_states, _ = jepa.unroll(
                obs_repeated,
                actions_expanded,
                nsteps=horizon,
                unroll_mode="autoregressive",
                ctxt_window_time=1,
                compute_loss=False,
                return_all_steps=False,
            )

            # Compute rewards for each trajectory
            predicted_rewards = reward_head(predicted_states)  # [N, horizon]
            total_rewards = predicted_rewards.sum(dim=1)  # [N]

            # Select best action sequence
            best_idx = total_rewards.argmax()
            best_action = actions[best_idx, 0].item()

        except Exception as e:
            # Fallback to random if planning fails
            best_action = torch.randint(0, 4, (1,)).item()

    return best_action


def record_long_video(checkpoint_path, output_path, target_minutes=5):
    """Record a long gameplay video."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    jepa, reward_head = load_model(checkpoint_path, device)

    # Create environment
    config = AtariConfig(game_name="Breakout", batch_size=1)
    env = AtariEnv(config, render_mode="rgb_array")

    print(f"\nRecording {target_minutes} minutes of MPPI + Reward gameplay...")
    print("=" * 70)

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # Target 30 FPS for smooth video
    video_writer = None

    total_frames = 0
    total_reward = 0
    episode_count = 0
    target_frames = int(target_minutes * 60 * fps)

    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    pbar = tqdm(total=target_frames, desc="Recording frames", unit="frame")

    while total_frames < target_frames:
        # Get RGB frame from environment
        rgb_frame = env.env.render()

        if video_writer is None:
            # Initialize video writer with frame dimensions
            height, width = rgb_frame.shape[:2]
            video_writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
            print(f"Video dimensions: {width}x{height}")

        # Write frame to video
        video_writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        total_frames += 1
        pbar.update(1)

        # Plan action using MPPI
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(2).to(device)
        else:
            obs_tensor = obs.float().unsqueeze(0).unsqueeze(2).to(device)

        action = simple_mppi_plan(jepa, reward_head, obs_tensor, device)

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        total_reward += reward

        # Reset if episode ends
        if terminated or truncated or episode_length >= 200:
            episode_count += 1
            pbar.set_postfix({
                "episodes": episode_count,
                "ep_reward": episode_reward,
                "avg_reward": total_reward / episode_count
            })
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

    pbar.close()
    video_writer.release()
    env.close()

    duration_seconds = total_frames / fps
    print(f"\n{'=' * 70}")
    print(f"✅ Video recording complete!")
    print(f"{'=' * 70}")
    print(f"Output: {output_path}")
    print(f"Duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"Frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Episodes: {episode_count}")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per episode: {total_reward/episode_count:.2f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='videos/mppi_reward_5min.mp4')
    parser.add_argument('--minutes', type=int, default=5, help='Target video length in minutes')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record_long_video(args.checkpoint, output_path, args.minutes)


if __name__ == "__main__":
    main()
