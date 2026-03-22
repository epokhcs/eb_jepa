"""
Visualize ATARI Breakout predictions from trained JEPA model.

Shows:
1. Ground truth frames
2. Predicted vs actual latent states (MSE)
3. Action sequences
4. Prediction quality over time
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA
from eb_jepa.logging import get_logger
from eb_jepa.training_utils import setup_device

logger = get_logger(__name__)


def load_model_and_data(checkpoint_path: str, config_path: str):
    """Load trained model and validation data."""
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

    # Load data
    logger.info("Loading validation data...")
    train_loader, val_loader = init_data(cfg.data.env_name, cfg.data)

    # Load checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model from checkpoint
    jepa = checkpoint['model']
    jepa.to(device)
    jepa.eval()

    return jepa, val_loader, device, cfg


def predict_sequence(jepa, x, a, device):
    """
    Predict future latent states given initial observation and actions.

    Args:
        jepa: Trained JEPA model
        x: Observations [B, C, T, H, W]
        a: Actions [B, 1, T]
        device: Device

    Returns:
        predicted_states: [B, D, T, H', W'] predicted latent states
        ground_truth_states: [B, D, T, H', W'] actual latent states
    """
    with torch.no_grad():
        # Encode ground truth sequence
        B, C, T, H, W = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2)  # [B*T, C, 1, H, W]
        gt_latents = jepa.encode(x_flat).squeeze(2)  # [B*T, D, H', W']
        D, H_lat, W_lat = gt_latents.shape[1:]
        ground_truth_states = gt_latents.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)  # [B, D, T, H', W']

        # Predict from first frame
        obs_init = x[:, :, 0:1]  # [B, C, 1, H, W]
        predicted_states = jepa.unroll(
            obs_init,
            a,
            nsteps=T-1,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=False,
            return_all_steps=False,
        )[0]  # [B, D, T-1, H', W']

        # Pad to match ground truth length
        # Ground truth has T frames, predictions has T-1 (no prediction for first frame)
        first_latent = ground_truth_states[:, :, 0:1]  # [B, D, 1, H', W']
        predicted_states = torch.cat([first_latent, predicted_states], dim=2)  # [B, D, T, H', W']

    return predicted_states, ground_truth_states


def compute_metrics(predicted, ground_truth):
    """Compute prediction quality metrics."""
    # MSE per timestep
    mse = ((predicted - ground_truth) ** 2).mean(dim=(1, 3, 4))  # [B, T]

    # Average over batch
    mean_mse = mse.mean(dim=0).cpu().numpy()  # [T]
    std_mse = mse.std(dim=0).cpu().numpy()  # [T]

    return mean_mse, std_mse


def visualize_predictions(
    jepa,
    val_loader,
    device,
    output_dir: Path,
    num_sequences: int = 5,
    max_horizon: int = 16,
):
    """Generate visualizations of model predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_mean_mse = []
    all_std_mse = []
    all_frames = []
    all_actions = []

    logger.info(f"Generating predictions for {num_sequences} sequences...")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, total=num_sequences)):
            if idx >= num_sequences:
                break

            # Extract batch
            if hasattr(batch, '_fields'):  # TrajectoryBatch
                x = batch.states.to(device)
                a = batch.actions.to(device)
            else:
                x, a = batch[0].to(device), batch[1].to(device)

            # Limit to max_horizon
            if x.shape[2] > max_horizon:
                x = x[:, :, :max_horizon]
                a = a[:, :, :max_horizon]

            # Predict
            predicted, ground_truth = predict_sequence(jepa, x, a, device)

            # Compute metrics
            mean_mse, std_mse = compute_metrics(predicted, ground_truth)
            all_mean_mse.append(mean_mse)
            all_std_mse.append(std_mse)

            # Save frames and actions for visualization
            # Denormalize frames [0, 1] -> [0, 255]
            normalizer = val_loader.dataset.normalizer
            frames = normalizer.unnormalize_state(x[0]).cpu().numpy()  # [C, T, H, W]
            frames = (frames * 255).clip(0, 255).astype(np.uint8)

            if frames.shape[0] == 1:  # Grayscale
                frames = frames[0]  # [T, H, W]
            else:  # RGB
                frames = frames.transpose(1, 2, 3, 0)  # [T, H, W, C]

            all_frames.append(frames)
            all_actions.append(a[0].cpu().numpy())  # [1, T]

    # Average metrics across sequences
    avg_mean_mse = np.mean(all_mean_mse, axis=0)
    avg_std_mse = np.mean(all_std_mse, axis=0)

    # Plot prediction error over time
    plot_prediction_error(avg_mean_mse, avg_std_mse, output_dir)

    # Create frame visualizations
    create_frame_visualizations(all_frames, all_actions, avg_mean_mse, output_dir)

    logger.info(f"✅ Visualizations saved to: {output_dir}")


def plot_prediction_error(mean_mse, std_mse, output_dir):
    """Plot prediction error over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    timesteps = np.arange(len(mean_mse))
    ax.plot(timesteps, mean_mse, 'b-', linewidth=2, label='Mean MSE')
    ax.fill_between(
        timesteps,
        mean_mse - std_mse,
        mean_mse + std_mse,
        alpha=0.3,
        color='b',
        label='±1 std'
    )

    ax.set_xlabel('Prediction Horizon (timesteps)', fontsize=12)
    ax.set_ylabel('Latent MSE', fontsize=12)
    ax.set_title('JEPA Prediction Quality Over Time (ATARI Breakout)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text annotation
    final_mse = mean_mse[-1]
    ax.text(
        0.95, 0.95,
        f'Final MSE (t={len(mean_mse)-1}): {final_mse:.3f}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_error.png', dpi=150)
    plt.close()

    logger.info(f"Saved prediction error plot: {output_dir / 'prediction_error.png'}")


def create_frame_visualizations(all_frames, all_actions, mean_mse, output_dir):
    """Create visualizations showing frames with prediction quality."""
    num_sequences = min(len(all_frames), 5)

    for seq_idx in range(num_sequences):
        frames = all_frames[seq_idx]  # [T, H, W] or [T, H, W, C]
        actions = all_actions[seq_idx][0]  # [T]

        # Create figure with frames and MSE overlay
        T = min(len(frames), 8)  # Show first 8 timesteps
        fig, axes = plt.subplots(2, T, figsize=(2*T, 4))

        for t in range(T):
            # Show frame
            ax_frame = axes[0, t]
            if frames.ndim == 3:  # Grayscale
                ax_frame.imshow(frames[t], cmap='gray', vmin=0, vmax=255)
            else:  # RGB
                ax_frame.imshow(frames[t])

            ax_frame.axis('off')

            # Action annotation
            action_val = int(actions[t])
            action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            action_name = action_names[action_val] if action_val < len(action_names) else str(action_val)
            ax_frame.set_title(f't={t}\nAction: {action_name}', fontsize=8)

            # MSE bar
            ax_mse = axes[1, t]
            mse_val = mean_mse[t] if t < len(mean_mse) else 0
            color = 'green' if mse_val < 0.2 else 'orange' if mse_val < 0.5 else 'red'
            ax_mse.bar([0], [mse_val], color=color, width=0.8)
            ax_mse.set_ylim([0, max(mean_mse) * 1.1])
            ax_mse.set_xlim([-0.5, 0.5])
            ax_mse.set_xticks([])
            ax_mse.set_ylabel('MSE', fontsize=8)
            ax_mse.text(0, mse_val + 0.01, f'{mse_val:.3f}', ha='center', fontsize=7)
            ax_mse.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Sequence {seq_idx+1}: Ground Truth Frames & Prediction Quality',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'sequence_{seq_idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved sequence visualization: {output_dir / f'sequence_{seq_idx+1}.png'}")


def create_video_from_frames(frames, output_path, fps=4):
    """Create MP4 video from frames."""
    try:
        import cv2

        if frames.ndim == 3:  # Grayscale [T, H, W]
            frames = np.stack([frames] * 3, axis=-1)  # [T, H, W, 3]

        T, H, W, C = frames.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

        for t in range(T):
            frame = frames[t]
            if C == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        logger.info(f"Saved video: {output_path}")

    except ImportError:
        logger.warning("opencv-python not installed, skipping video generation")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize JEPA predictions on ATARI Breakout')
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
        '--output_dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--num_sequences',
        type=int,
        default=5,
        help='Number of sequences to visualize'
    )
    parser.add_argument(
        '--max_horizon',
        type=int,
        default=16,
        help='Maximum prediction horizon'
    )

    args = parser.parse_args()

    # Load model and data
    jepa, val_loader, device, cfg = load_model_and_data(args.checkpoint, args.config)

    # Generate visualizations
    output_dir = Path(args.output_dir)
    visualize_predictions(
        jepa,
        val_loader,
        device,
        output_dir,
        num_sequences=args.num_sequences,
        max_horizon=args.max_horizon,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Visualization complete!")
    logger.info(f"{'='*60}")
    logger.info(f"📊 View results in: {output_dir.absolute()}")
    logger.info(f"   - prediction_error.png: MSE over time")
    logger.info(f"   - sequence_*.png: Frame sequences with prediction quality")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
