"""
Test reward prediction accuracy on validation data.

This script:
1. Loads a trained JEPA model with reward head
2. Evaluates reward prediction MSE on validation trajectories
3. Visualizes predicted vs actual rewards
4. Compares reward prediction quality across actions
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from eb_jepa.datasets.utils import init_data
from eb_jepa.utils import init_model
from eb_jepa.architectures import RewardPredictionHead


def test_reward_prediction(checkpoint_path, cfg, device, num_batches=10):
    """Test reward prediction accuracy on validation data."""

    # Load data
    print("Loading validation data...")
    _, val_loader = init_data(
        env_name=cfg.data.env_name,
        cfg_data=cfg.data,
        batch_size=cfg.optimization.batch_size,
        num_workers=0,
        pin_memory=False
    )

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize JEPA
    jepa = init_model(
        encoder_arch=cfg.model.encoder_architecture,
        predictor_arch=cfg.model.predictor_architecture,
        dobs=cfg.data.dobs,
        henc=cfg.model.henc,
        hpre=cfg.model.hpre,
        dstc=cfg.model.dstc,
        action_dim=1,  # ATARI discrete actions
        compile_model=False,
    ).to(device)
    jepa.load_state_dict(checkpoint['model'], strict=False)
    jepa.eval()

    # Initialize reward head
    print("Loading reward prediction head...")
    state_dim = cfg.model.dstc
    reward_head = RewardPredictionHead(
        state_dim=state_dim,
        hidden_dim=cfg.model.get("reward_head_hidden_dim", 256),
        spatial_aggregate="mean",
    ).to(device)

    # Load reward head weights
    if 'reward_head' in checkpoint:
        reward_head.load_state_dict(checkpoint['reward_head'])
        print("✅ Loaded reward head from checkpoint")
    else:
        print("⚠️  No reward head in checkpoint, using random initialization")

    # Collect predictions
    all_predicted_rewards = []
    all_actual_rewards = []
    all_actions = []

    print(f"\nEvaluating on {num_batches} batches...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break

            x = batch.states.to(device)
            a = batch.actions.to(device)

            # Get predicted latent states
            predicted_latents, _ = jepa.unroll(
                x, a,
                nsteps=cfg.model.nsteps,
                unroll_mode="autoregressive",
                ctxt_window_time=1,
                compute_loss=False,
                return_all_steps=False,
            )

            # Predict rewards
            predicted_rewards = reward_head(predicted_latents)  # [B, T]

            # Get actual rewards
            if hasattr(batch, 'metadata') and 'rewards' in batch.metadata:
                actual_rewards = batch.metadata['rewards'][:, :predicted_latents.shape[2]]
            else:
                print("⚠️  No reward metadata in batch, skipping")
                continue

            # Store results
            all_predicted_rewards.append(predicted_rewards.cpu().numpy())
            all_actual_rewards.append(actual_rewards.cpu().numpy())
            all_actions.append(a.cpu().numpy())

            if i % 5 == 0:
                mse = ((predicted_rewards.cpu() - actual_rewards.cpu()) ** 2).mean().item()
                print(f"  Batch {i}: MSE = {mse:.4f}")

    # Concatenate all results
    all_predicted_rewards = np.concatenate(all_predicted_rewards, axis=0)  # [N, T]
    all_actual_rewards = np.concatenate(all_actual_rewards, axis=0)  # [N, T]
    all_actions = np.concatenate(all_actions, axis=0)  # [N, 1, T]

    # Calculate overall metrics
    print("\n" + "="*60)
    print("REWARD PREDICTION RESULTS")
    print("="*60)

    # Overall MSE
    overall_mse = ((all_predicted_rewards - all_actual_rewards) ** 2).mean()
    overall_mae = np.abs(all_predicted_rewards - all_actual_rewards).mean()
    print(f"\n📊 Overall Metrics:")
    print(f"  MSE:  {overall_mse:.4f}")
    print(f"  MAE:  {overall_mae:.4f}")
    print(f"  RMSE: {np.sqrt(overall_mse):.4f}")

    # Per-timestep MSE
    print(f"\n📈 MSE by Timestep:")
    timestep_mse = ((all_predicted_rewards - all_actual_rewards) ** 2).mean(axis=0)
    for t in range(min(8, len(timestep_mse))):
        print(f"  t={t}: {timestep_mse[t]:.4f}")

    # Per-action MSE
    print(f"\n🎮 MSE by Action:")
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    for action_idx in range(4):
        mask = (all_actions[:, 0, :] == action_idx).flatten()
        if mask.sum() > 0:
            action_mse = ((all_predicted_rewards.flatten()[mask] -
                          all_actual_rewards.flatten()[mask]) ** 2).mean()
            print(f"  {action_names[action_idx]:>6s}: {action_mse:.4f}")

    # Visualize predictions
    print(f"\n📊 Generating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Scatter plot: Predicted vs Actual
    ax = axes[0, 0]
    sample_indices = np.random.choice(len(all_predicted_rewards.flatten()),
                                     size=min(1000, len(all_predicted_rewards.flatten())),
                                     replace=False)
    ax.scatter(all_actual_rewards.flatten()[sample_indices],
              all_predicted_rewards.flatten()[sample_indices],
              alpha=0.3, s=10)
    ax.plot([all_actual_rewards.min(), all_actual_rewards.max()],
           [all_actual_rewards.min(), all_actual_rewards.max()],
           'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Reward')
    ax.set_ylabel('Predicted Reward')
    ax.set_title(f'Reward Prediction Accuracy\n(MSE: {overall_mse:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MSE by timestep
    ax = axes[0, 1]
    ax.plot(range(len(timestep_mse)), timestep_mse, marker='o')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('MSE')
    ax.set_title('Prediction Error Over Time')
    ax.grid(True, alpha=0.3)

    # 3. Example trajectories
    ax = axes[1, 0]
    for i in range(min(5, len(all_predicted_rewards))):
        ax.plot(all_actual_rewards[i], alpha=0.6, linestyle='--', label=f'Actual {i}')
        ax.plot(all_predicted_rewards[i], alpha=0.8, label=f'Pred {i}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Reward')
    ax.set_title('Example Reward Trajectories')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. MSE by action
    ax = axes[1, 1]
    action_mses = []
    for action_idx in range(4):
        mask = (all_actions[:, 0, :] == action_idx).flatten()
        if mask.sum() > 0:
            action_mse = ((all_predicted_rewards.flatten()[mask] -
                          all_actual_rewards.flatten()[mask]) ** 2).mean()
            action_mses.append(action_mse)
        else:
            action_mses.append(0)
    ax.bar(action_names, action_mses)
    ax.set_ylabel('MSE')
    ax.set_title('Prediction Error by Action')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    save_path = Path(checkpoint_path).parent / "reward_prediction_eval.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to {save_path}")

    plt.show()

    return {
        'mse': overall_mse,
        'mae': overall_mae,
        'timestep_mse': timestep_mse,
    }


def main():
    parser = argparse.ArgumentParser(description='Test reward prediction accuracy')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str,
                       default='examples/ac_video_jepa/cfgs/train_atari.yaml',
                       help='Path to config file')
    parser.add_argument('--num_batches', type=int, default=10,
                       help='Number of validation batches to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu/mps)')

    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Run test
    results = test_reward_prediction(args.checkpoint, cfg, device, args.num_batches)

    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. If MSE < 0.5: Reward prediction is working well! 🎉")
    print(f"2. Update test_planning.py to use reward-based objective")
    print(f"3. Run planning comparison: latent variance vs predicted rewards")
    print(f"4. Expected: 5-10x better game scores with reward optimization")


if __name__ == '__main__':
    main()
