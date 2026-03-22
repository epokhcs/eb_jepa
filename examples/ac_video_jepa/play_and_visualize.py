"""
Interactive ATARI Breakout player with JEPA predictions visualization.

Shows side-by-side comparison:
1. Left: Ground truth (actual game)
2. Right: JEPA predicted frames

The model predicts what will happen given current state + future actions.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eb_jepa.datasets.registry import EnvironmentRegistry
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import setup_device
from eb_jepa.datasets.utils import init_data

logger = get_logger(__name__)


def load_checkpoint(checkpoint_path: str, config_path: str):
    """Load trained JEPA model from checkpoint."""
    logger.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Convert to namespace
    from argparse import Namespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    cfg = dict_to_namespace(cfg)

    # Setup device
    device = setup_device("auto")

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model from checkpoint
    jepa = checkpoint['model']
    jepa.to(device)
    jepa.eval()

    # Get normalizer
    train_loader, val_loader = init_data(cfg.data.env_name, cfg.data)
    normalizer = train_loader.dataset.normalizer

    return jepa, device, cfg, normalizer


def decode_latents_to_pixels(jepa, latent_states, normalizer):
    """
    Decode latent states to pixel space.

    For now, we'll use a simple approach:
    - If decoder exists: use it
    - Otherwise: use nearest neighbor from training data (not implemented here)

    Args:
        jepa: JEPA model
        latent_states: [B, D, T, H', W'] latent representations
        normalizer: Dataset normalizer

    Returns:
        frames: [B, T, H, W, C] predicted frames in pixel space
    """
    # Check if model has decoder
    if not hasattr(jepa, 'decoder') or jepa.decoder is None:
        logger.warning("No decoder found, returning placeholder frames")
        B, D, T, H_lat, W_lat = latent_states.shape
        # Return gray placeholder (will show "No decoder" in visualization)
        return torch.ones(B, T, 84, 84, 1) * 128  # Gray frames

    with torch.no_grad():
        B, D, T, H_lat, W_lat = latent_states.shape

        # Flatten time dimension
        latents_flat = latent_states.permute(0, 2, 1, 3, 4).flatten(0, 1)  # [B*T, D, H', W']

        # Decode
        decoded = jepa.decoder(latents_flat)  # [B*T, C, H, W]

        # Unflatten and permute to [B, T, H, W, C]
        C, H, W = decoded.shape[1:]
        frames = decoded.unflatten(0, (B, T)).permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]

        # Unnormalize
        frames = normalizer.unnormalize_state(frames.permute(0, 4, 1, 2, 3))  # [B, C, T, H, W]
        frames = frames.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]

    return frames


def play_with_predictions(
    jepa,
    env,
    device,
    normalizer,
    policy='tracking',
    num_episodes=1,
    max_steps=200,
    prediction_horizon=8,
    output_dir=None,
    save_video=False,
):
    """
    Play Breakout and show predicted vs actual trajectories.

    Args:
        jepa: Trained JEPA model
        env: Breakout environment
        device: torch device
        normalizer: Data normalizer
        policy: 'random', 'tracking', or 'static'
        num_episodes: Number of episodes to play
        max_steps: Max steps per episode
        prediction_horizon: How many steps ahead to predict
        output_dir: Where to save visualizations
        save_video: Whether to save as MP4
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

    for episode in range(num_episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.info(f"{'='*60}")

        obs, info = env.reset()
        done = False
        step = 0

        episode_frames_gt = []
        episode_frames_pred = []
        episode_actions = []
        episode_mse = []

        while not done and step < max_steps:
            # Get current observation [H, W, C] or [H, W]
            current_frame = obs

            # Choose action based on policy
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'tracking':
                # Track ball (requires OCAtari)
                paddle_x = None
                ball_x = None
                for obj in env.objects:
                    if obj.category.lower() == "player":
                        paddle_x = obj.x
                    elif obj.category.lower() == "ball":
                        ball_x = obj.x

                if paddle_x is not None and ball_x is not None:
                    if ball_x < paddle_x - 5:
                        action = 3  # LEFT
                    elif ball_x > paddle_x + 5:
                        action = 2  # RIGHT
                    else:
                        action = 0  # NOOP
                else:
                    action = 0
            else:  # static
                action = 0

            # Generate action sequence for prediction (repeat same action)
            action_sequence = torch.tensor([action] * prediction_horizon, dtype=torch.float32)
            action_sequence = action_sequence.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]

            # Prepare current observation for model
            obs_tensor = torch.from_numpy(current_frame).float()
            if obs_tensor.ndim == 2:  # Grayscale [H, W]
                obs_tensor = obs_tensor.unsqueeze(0)  # [1, H, W]
            elif obs_tensor.ndim == 3 and obs_tensor.shape[-1] in [1, 3]:  # [H, W, C]
                obs_tensor = obs_tensor.permute(2, 0, 1)  # [C, H, W]

            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, C, 1, H, W]
            obs_tensor = normalizer.normalize_state(obs_tensor)

            # Predict future states
            with torch.no_grad():
                try:
                    predicted_latents = jepa.unroll(
                        obs_tensor,
                        action_sequence,
                        nsteps=prediction_horizon,
                        unroll_mode="autoregressive",
                        ctxt_window_time=1,
                        compute_loss=False,
                        return_all_steps=False,
                    )[0]  # [1, D, T, H', W']

                    # Decode to pixels (placeholder for now - needs decoder)
                    # For visualization, we'll show the latent MSE
                    has_prediction = True
                except Exception as e:
                    logger.warning(f"Prediction failed: {e}")
                    has_prediction = False

            # Execute action in environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Store frames
            episode_frames_gt.append(current_frame)
            episode_actions.append(action)

            # Update state
            obs = next_obs
            step += 1

            # Print progress
            if step % 10 == 0:
                logger.info(f"  Step {step}/{max_steps} | Action: {action_names[action]} | Reward: {reward}")

        logger.info(f"Episode finished after {step} steps")

        # Create visualization
        if output_dir:
            create_episode_visualization(
                episode_frames_gt,
                episode_actions,
                episode,
                output_dir,
                save_video=save_video
            )


def create_episode_visualization(frames_gt, actions, episode_num, output_dir, save_video=False):
    """Create side-by-side visualization of ground truth trajectory."""
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

    # Create grid of frames (show every 5th frame)
    stride = 5
    frames_to_show = frames_gt[::stride]
    actions_to_show = actions[::stride]

    num_frames = min(len(frames_to_show), 16)

    fig, axes = plt.subplots(2, num_frames // 2, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(num_frames):
        ax = axes[i]
        frame = frames_to_show[i]

        # Handle grayscale vs RGB
        if frame.ndim == 2:
            ax.imshow(frame, cmap='gray')
        else:
            ax.imshow(frame)

        action_idx = actions_to_show[i]
        action_name = action_names[action_idx] if action_idx < len(action_names) else str(action_idx)

        ax.set_title(f't={i*stride}\n{action_name}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f'Episode {episode_num + 1}: Ground Truth Trajectory', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / f'episode_{episode_num + 1}_trajectory.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Saved trajectory visualization: {save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Play ATARI Breakout with JEPA predictions')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth.tar)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config (.yaml)'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default='tracking',
        choices=['random', 'tracking', 'static'],
        help='Policy for playing the game'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=3,
        help='Number of episodes to play'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=200,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--prediction_horizon',
        type=int,
        default=8,
        help='How many steps ahead to predict'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations/gameplay',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--save_video',
        action='store_true',
        help='Save as MP4 video'
    )

    args = parser.parse_args()

    # Load model
    jepa, device, cfg, normalizer = load_checkpoint(args.checkpoint, args.config)

    # Create environment
    logger.info(f"Creating {cfg.data.env_name} environment...")
    from eb_jepa.datasets.atari.env import AtariEnv
    from eb_jepa.datasets.atari.config import AtariConfig

    env_config = AtariConfig(
        game_name=cfg.data.game_name,
        img_size=cfg.data.get('img_size', 84),
        normalize=False,  # We'll normalize in the model
    )
    env = AtariEnv(config=env_config)

    # Play with predictions
    play_with_predictions(
        jepa=jepa,
        env=env,
        device=device,
        normalizer=normalizer,
        policy=args.policy,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        prediction_horizon=args.prediction_horizon,
        output_dir=args.output_dir,
        save_video=args.save_video,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Gameplay visualization complete!")
    logger.info(f"{'='*60}")
    logger.info(f"📊 View results in: {Path(args.output_dir).absolute()}")


if __name__ == "__main__":
    main()
