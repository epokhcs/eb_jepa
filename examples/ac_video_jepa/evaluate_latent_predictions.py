"""
Evaluate JEPA prediction quality in latent space without decoder.

Strategy:
1. Play Breakout and record real trajectories (frames + actions)
2. For each initial state, rollout predictions
3. Encode both predicted and actual frames to latent space
4. Compute cosine similarity between predicted and actual latents
5. Plot similarity degradation over prediction horizon

This measures how well the model understands game dynamics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eb_jepa.checkpoint_utils import load_jepa_from_checkpoint
from eb_jepa.datasets.registry import EnvironmentRegistry
from eb_jepa.datasets.atari.config import AtariConfig
from eb_jepa.logging import get_logger

logger = get_logger(__name__)


def record_gameplay_trajectories(env, policy='tracking', num_episodes=10, max_steps=200):
    """
    Record real gameplay trajectories with frames and actions.

    Args:
        env: Breakout environment
        policy: 'random' or 'tracking'
        num_episodes: Number of episodes to record
        max_steps: Max steps per episode

    Returns:
        trajectories: List of dicts with 'frames', 'actions'
    """
    logger.info(f"Recording {num_episodes} gameplay episodes with {policy} policy...")

    trajectories = []

    for episode in tqdm(range(num_episodes), desc="Recording episodes"):
        obs, info = env.reset()
        frames = [obs]
        actions = []

        step = 0
        done = False

        while not done and step < max_steps:
            # Choose action based on policy
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'tracking':
                # Track ball
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
            else:
                action = 0  # static

            # Execute action
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            frames.append(next_obs)
            actions.append(action)

            obs = next_obs
            step += 1

        trajectories.append({
            'frames': np.array(frames),  # [T+1, H, W] or [T+1, H, W, C]
            'actions': np.array(actions),  # [T]
        })

    logger.info(f"Recorded {len(trajectories)} trajectories, avg length: {np.mean([len(t['actions']) for t in trajectories]):.1f}")
    return trajectories


def evaluate_prediction_quality(
    jepa,
    trajectories,
    normalizer,
    device,
    prediction_horizons=[1, 2, 4, 8, 16],
    num_samples=100,
):
    """
    Evaluate prediction quality by comparing predicted vs actual latents.

    Args:
        jepa: Trained JEPA model
        trajectories: List of recorded gameplay trajectories
        normalizer: Data normalizer
        device: torch device
        prediction_horizons: List of horizons to evaluate
        num_samples: Number of initial states to sample

    Returns:
        results: Dict with cosine similarities per horizon
    """
    logger.info(f"Evaluating prediction quality for {num_samples} samples...")

    max_horizon = max(prediction_horizons)

    # Collect samples (initial state + future trajectory)
    samples = []
    for traj in trajectories:
        frames = traj['frames']
        actions = traj['actions']
        T = len(actions)

        # Sample multiple starting points from this trajectory
        for t0 in range(0, T - max_horizon, max_horizon // 2):
            if t0 + max_horizon >= T:
                continue

            samples.append({
                'init_frame': frames[t0],
                'future_frames': frames[t0+1:t0+max_horizon+1],
                'actions': actions[t0:t0+max_horizon],
            })

            if len(samples) >= num_samples:
                break

        if len(samples) >= num_samples:
            break

    samples = samples[:num_samples]
    logger.info(f"Collected {len(samples)} samples for evaluation")

    # Evaluate each sample
    cosine_similarities = {h: [] for h in prediction_horizons}

    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating samples"):
            init_frame = sample['init_frame']
            future_frames = sample['future_frames']
            actions = sample['actions']

            # Prepare initial observation
            obs_tensor = torch.from_numpy(init_frame).float()
            if obs_tensor.ndim == 2:  # Grayscale [H, W]
                obs_tensor = obs_tensor.unsqueeze(0)  # [1, H, W]
            elif obs_tensor.ndim == 3 and obs_tensor.shape[-1] in [1, 3]:  # [H, W, C]
                obs_tensor = obs_tensor.permute(2, 0, 1)  # [C, H, W]

            obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(2).to(device)  # [1, C, 1, H, W]
            obs_tensor = normalizer.normalize_state(obs_tensor)

            # Prepare actions
            action_tensor = torch.from_numpy(actions).float()
            action_tensor = action_tensor.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T]

            # Predict future latent states
            predicted_latents = jepa.unroll(
                obs_tensor,
                action_tensor,
                nsteps=len(actions),
                unroll_mode="autoregressive",
                ctxt_window_time=1,
                compute_loss=False,
                return_all_steps=False,
            )[0]  # [1, D, T, H', W']

            # Encode actual future frames
            actual_frames_list = []
            for frame in future_frames:
                frame_tensor = torch.from_numpy(frame).float()
                if frame_tensor.ndim == 2:
                    frame_tensor = frame_tensor.unsqueeze(0)
                elif frame_tensor.ndim == 3 and frame_tensor.shape[-1] in [1, 3]:
                    frame_tensor = frame_tensor.permute(2, 0, 1)

                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(2).to(device)
                frame_tensor = normalizer.normalize_state(frame_tensor)
                actual_frames_list.append(frame_tensor)

            actual_frames_tensor = torch.cat(actual_frames_list, dim=2)  # [1, C, T, H, W]

            # Encode actual frames
            B, C, T, H, W = actual_frames_tensor.shape
            actual_frames_flat = actual_frames_tensor.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2)  # [B*T, C, 1, H, W]
            actual_latents = jepa.encode(actual_frames_flat).squeeze(2)  # [B*T, D, H', W']
            D, H_lat, W_lat = actual_latents.shape[1:]
            actual_latents = actual_latents.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)  # [1, D, T, H', W']

            # Compute cosine similarity for each horizon
            for h in prediction_horizons:
                if h > len(actions):
                    continue

                # Get latents at horizon h
                pred_h = predicted_latents[:, :, h-1]  # [1, D, H', W']
                actual_h = actual_latents[:, :, h-1]  # [1, D, H', W']

                # Flatten spatial dimensions
                pred_flat = pred_h.flatten(1)  # [1, D*H'*W']
                actual_flat = actual_h.flatten(1)  # [1, D*H'*W']

                # Compute cosine similarity
                cos_sim = F.cosine_similarity(pred_flat, actual_flat, dim=1)
                cosine_similarities[h].append(cos_sim.item())

    # Aggregate results
    results = {
        'horizons': prediction_horizons,
        'mean_cosine_sim': [np.mean(cosine_similarities[h]) for h in prediction_horizons],
        'std_cosine_sim': [np.std(cosine_similarities[h]) for h in prediction_horizons],
        'all_similarities': cosine_similarities,
    }

    return results


def plot_results(results, output_path):
    """Plot cosine similarity vs prediction horizon."""
    horizons = results['horizons']
    mean_sim = results['mean_cosine_sim']
    std_sim = results['std_cosine_sim']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(horizons, mean_sim, 'b-', linewidth=2, marker='o', markersize=8, label='Mean Cosine Similarity')
    ax.fill_between(
        horizons,
        np.array(mean_sim) - np.array(std_sim),
        np.array(mean_sim) + np.array(std_sim),
        alpha=0.3,
        color='b',
        label='±1 std'
    )

    ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% similarity')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80% similarity')
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='70% similarity')

    ax.set_xlabel('Prediction Horizon (steps)', fontsize=12)
    ax.set_ylabel('Cosine Similarity (Latent Space)', fontsize=12)
    ax.set_title('JEPA Prediction Quality: Latent Space Similarity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Add text annotation
    final_sim = mean_sim[-1]
    ax.text(
        0.95, 0.05,
        f'Final similarity (h={horizons[-1]}): {final_sim:.3f}\n'
        f'Model prediction horizon: ~{horizons[np.argmax(np.array(mean_sim) < 0.8)]} steps',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"✅ Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate JEPA prediction quality in latent space')
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
        help='Policy for recording gameplay'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='Number of episodes to record'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='Number of initial states to evaluate'
    )
    parser.add_argument(
        '--max_horizon',
        type=int,
        default=16,
        help='Maximum prediction horizon to evaluate'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations/latent_eval',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("=" * 60)
    logger.info("LOADING MODEL")
    logger.info("=" * 60)
    jepa, cfg, data_config = load_jepa_from_checkpoint(args.checkpoint, args.config, device='auto')
    device = next(jepa.parameters()).device

    # Get normalizer
    from eb_jepa.datasets.utils import init_data
    _, val_loader, _ = init_data(cfg.data.env_name, vars(cfg.data))
    normalizer = val_loader.dataset.normalizer

    # Create environment
    logger.info("=" * 60)
    logger.info("CREATING ENVIRONMENT")
    logger.info("=" * 60)
    from eb_jepa.datasets.atari.env import AtariEnv

    env_config = AtariConfig(
        game_name=cfg.data.game_name,
        img_size=data_config.img_size,
        normalize=False,
    )
    env = AtariEnv(config=env_config)

    # Record gameplay trajectories
    logger.info("=" * 60)
    logger.info("RECORDING GAMEPLAY")
    logger.info("=" * 60)
    trajectories = record_gameplay_trajectories(
        env=env,
        policy=args.policy,
        num_episodes=args.num_episodes,
        max_steps=200
    )

    # Evaluate prediction quality
    logger.info("=" * 60)
    logger.info("EVALUATING PREDICTIONS")
    logger.info("=" * 60)
    prediction_horizons = list(range(1, args.max_horizon + 1))
    results = evaluate_prediction_quality(
        jepa=jepa,
        trajectories=trajectories,
        normalizer=normalizer,
        device=device,
        prediction_horizons=prediction_horizons,
        num_samples=args.num_samples,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    for h, mean, std in zip(results['horizons'], results['mean_cosine_sim'], results['std_cosine_sim']):
        logger.info(f"Horizon {h:2d}: {mean:.4f} ± {std:.4f}")

    # Plot results
    logger.info("=" * 60)
    logger.info("PLOTTING")
    logger.info("=" * 60)
    plot_results(results, output_dir / 'latent_similarity.png')

    # Save raw results
    import pickle
    with open(output_dir / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"✅ Saved results: {output_dir / 'results.pkl'}")

    logger.info("=" * 60)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"📊 View results in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
