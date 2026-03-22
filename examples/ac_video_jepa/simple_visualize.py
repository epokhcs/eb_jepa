"""
Simple visualization script for JEPA predictions on ATARI.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--num_sequences', type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Rebuild model (we need to know the architecture)
    from eb_jepa.architectures import ImpalaEncoder, RNNPredictor, DiscreteActionEncoder
    from eb_jepa.losses import VC_IDM_Sim_Regularizer, SquareLossSeq
    from eb_jepa.jepa import JEPA
    import torch.nn as nn

    # Model config (from train_atari.yaml)
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, 256, 256),  # henc=256, dstc=256
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=1,  # grayscale
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

    # Create regularizer (with dummy IDM for now)
    from eb_jepa.architectures import DiscreteInverseDynamicsModel
    from eb_jepa.losses import DiscreteInverseDynamicsLoss

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
    print(f"✅ Model loaded successfully")

    # Create dataset
    from eb_jepa.datasets.atari.dataset import AtariDataset
    from eb_jepa.datasets.atari.config import AtariConfig

    config = AtariConfig(
        game_name="Breakout",
        batch_size=4,
        size=100,
        val_size=20,
        n_steps=200,
        sample_length=17,
        grayscale=True,
        img_size=84,
        normalize=True,
    )

    print("Creating validation dataset...")
    val_dataset = AtariDataset(config)
    print(f"✅ Dataset created with {len(val_dataset)} samples")

    # Collect predictions
    all_mse = []
    all_actions = []

    print(f"\n🔮 Generating predictions for {args.num_sequences} sequences...")

    with torch.no_grad():
        for idx in tqdm(range(min(args.num_sequences, len(val_dataset)))):
            batch = val_dataset[idx]
            x = batch.states.unsqueeze(0).to(device)  # [1, C, T, H, W]
            a = batch.actions.unsqueeze(0).to(device)  # [1, 1, T]

            B, C, T, H, W = x.shape

            # Encode ground truth
            x_flat = x.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2)
            gt_latents = jepa.encode(x_flat).squeeze(2)
            D, H_lat, W_lat = gt_latents.shape[1:]
            ground_truth = gt_latents.unflatten(0, (B, T)).permute(0, 2, 1, 3, 4)

            # Predict from first frame
            obs_init = x[:, :, 0:1]

            # Limit actions to match prediction horizon
            a_pred = a[:, :, :T]

            predicted = jepa.unroll(
                obs_init,
                a_pred,
                nsteps=T-1,
                unroll_mode="autoregressive",
                ctxt_window_time=1,
                compute_loss=False,
                return_all_steps=False,
            )[0]

            # Prepend first latent to predictions
            first_latent = ground_truth[:, :, 0:1]
            predicted = torch.cat([first_latent, predicted], dim=2)

            # Ensure shapes match by trimming to minimum length
            min_len = min(predicted.shape[2], ground_truth.shape[2])
            predicted = predicted[:, :, :min_len]
            ground_truth = ground_truth[:, :, :min_len]

            # Compute MSE
            mse = ((predicted - ground_truth) ** 2).mean(dim=(1, 3, 4))
            all_mse.append(mse[0].cpu().numpy())

            # Store actions for analysis
            all_actions.append(a[0, 0, :min_len].cpu().numpy())

    # Average MSE
    avg_mse = np.mean(all_mse, axis=0)
    std_mse = np.std(all_mse, axis=0)

    # Convert to numpy arrays for easier indexing
    all_mse_array = np.array(all_mse)  # [num_sequences, T]
    all_actions_array = np.array(all_actions)  # [num_sequences, T]

    # Action names for Breakout
    action_names = {0: 'NOOP', 1: 'FIRE', 2: 'RIGHT', 3: 'LEFT'}
    action_colors = {0: 'gray', 1: 'red', 2: 'blue', 3: 'green'}

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Overall prediction error (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    timesteps = np.arange(len(avg_mse))
    ax1.plot(timesteps, avg_mse, 'b-', linewidth=2, label='Mean MSE')
    ax1.fill_between(timesteps, avg_mse - std_mse, avg_mse + std_mse, alpha=0.3)
    ax1.set_xlabel('Prediction Horizon (timesteps)')
    ax1.set_ylabel('Latent Space MSE')
    ax1.set_title('Overall Prediction Quality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: MSE by action type (top right)
    ax2 = fig.add_subplot(gs[0, 1])

    # Compute MSE conditioned on previous action
    action_mse = {action: [] for action in range(4)}
    for t in range(1, len(avg_mse)):  # Start from t=1 (first prediction)
        for action in range(4):
            # Find all sequences where action[t-1] == action
            mask = all_actions_array[:, t-1] == action
            if mask.sum() > 0:
                mse_for_action = all_mse_array[mask, t].mean()
                action_mse[action].append((t, mse_for_action))

    for action in range(4):
        if len(action_mse[action]) > 0:
            times, mses = zip(*action_mse[action])
            ax2.plot(times, mses, marker='o', linewidth=2,
                    label=f'{action_names[action]}',
                    color=action_colors[action], alpha=0.7)

    ax2.set_xlabel('Prediction Horizon (timesteps)')
    ax2.set_ylabel('Latent Space MSE')
    ax2.set_title('Prediction Quality by Previous Action')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Action distribution over time (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    action_counts = np.zeros((len(avg_mse), 4))
    for t in range(len(avg_mse)):
        for action in range(4):
            action_counts[t, action] = (all_actions_array[:, t] == action).sum()

    action_counts = action_counts / action_counts.sum(axis=1, keepdims=True) * 100
    bottom = np.zeros(len(avg_mse))
    for action in range(4):
        ax3.bar(timesteps, action_counts[:, action], bottom=bottom,
               label=action_names[action], color=action_colors[action], alpha=0.7)
        bottom += action_counts[:, action]

    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Action Distribution (%)')
    ax3.set_title('Action Distribution Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Average MSE by action (middle right) - bar chart
    ax4 = fig.add_subplot(gs[1, 1])
    avg_mse_by_action = []
    action_labels = []
    action_bar_colors = []
    for action in range(4):
        mses_for_action = []
        for t in range(1, len(avg_mse)):
            mask = all_actions_array[:, t-1] == action
            if mask.sum() > 0:
                mses_for_action.append(all_mse_array[mask, t].mean())
        if len(mses_for_action) > 0:
            avg_mse_by_action.append(np.mean(mses_for_action))
            action_labels.append(action_names[action])
            action_bar_colors.append(action_colors[action])

    bars = ax4.bar(action_labels, avg_mse_by_action, color=action_bar_colors, alpha=0.7)
    ax4.set_ylabel('Average MSE')
    ax4.set_title('Average Prediction Error by Action')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, avg_mse_by_action):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 5: Action transition heatmap (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    transition_matrix = np.zeros((4, 4))
    for seq_actions in all_actions_array:
        for t in range(len(seq_actions) - 1):
            action_t = int(seq_actions[t])
            action_t1 = int(seq_actions[t+1])
            transition_matrix[action_t, action_t1] += 1

    # Normalize rows
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums,
                                  where=row_sums!=0, out=np.zeros_like(transition_matrix))

    im = ax5.imshow(transition_matrix, cmap='Blues', aspect='auto')
    ax5.set_xticks(range(4))
    ax5.set_yticks(range(4))
    ax5.set_xticklabels([action_names[i] for i in range(4)])
    ax5.set_yticklabels([action_names[i] for i in range(4)])
    ax5.set_xlabel('Next Action')
    ax5.set_ylabel('Current Action')
    ax5.set_title('Action Transition Probabilities')

    # Add text annotations
    for i in range(4):
        for j in range(4):
            text = ax5.text(j, i, f'{transition_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax5)

    # Plot 6: Summary statistics (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    summary_text = f"""
    📊 PREDICTION QUALITY SUMMARY
    {'─'*40}

    Overall Statistics:
    • Initial MSE (t=0):  {avg_mse[0]:.4f}
    • Final MSE (t={len(avg_mse)-1}):    {avg_mse[-1]:.4f}
    • Average MSE:        {np.mean(avg_mse):.4f}
    • Std Dev:            {np.mean(std_mse):.4f}

    {'─'*40}

    MSE by Action (avg across time):
    """

    for action, label in action_names.items():
        mses_for_action = []
        for t in range(1, len(avg_mse)):
            mask = all_actions_array[:, t-1] == action
            if mask.sum() > 0:
                mses_for_action.append(all_mse_array[mask, t].mean())
        if len(mses_for_action) > 0:
            summary_text += f"• {label:6s}:  {np.mean(mses_for_action):.4f}\n    "

    summary_text += f"""

    {'─'*40}

    Dataset Info:
    • Sequences analyzed: {len(all_mse)}
    • Timesteps per seq:  {len(avg_mse)}
    • Total predictions:  {len(all_mse) * len(avg_mse)}
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('JEPA Prediction Quality Analysis on ATARI Breakout',
                fontsize=16, fontweight='bold', y=0.995)

    output_path = output_dir / 'prediction_error.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved comprehensive plot: {output_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"📊 Prediction Quality Summary")
    print(f"{'='*60}")
    print(f"Initial MSE (t=0): {avg_mse[0]:.4f}")
    print(f"Final MSE (t={len(avg_mse)-1}): {avg_mse[-1]:.4f}")
    print(f"Average MSE: {np.mean(avg_mse):.4f}")
    print(f"\nMSE by Action (average):")
    for action, label in action_names.items():
        mses_for_action = []
        for t in range(1, len(avg_mse)):
            mask = all_actions_array[:, t-1] == action
            if mask.sum() > 0:
                mses_for_action.append(all_mse_array[mask, t].mean())
        if len(mses_for_action) > 0:
            print(f"  {label}: {np.mean(mses_for_action):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
