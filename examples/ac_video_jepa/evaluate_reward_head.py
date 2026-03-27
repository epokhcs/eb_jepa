#!/usr/bin/env python3
"""
Evaluate the reward prediction head quality.

Tests how well the trained reward head can predict rewards from latent states.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from eb_jepa.checkpoint_utils import load_jepa_from_checkpoint
from eb_jepa.datasets.utils import init_data


def evaluate_reward_head(jepa, val_loader, device, num_batches=50):
    """
    Evaluate reward prediction accuracy on validation data.

    Args:
        jepa: JEPA model with reward head
        val_loader: Validation data loader
        device: Device to run on
        num_batches: Number of batches to evaluate

    Returns:
        dict: Results including accuracy, MSE, and per-class metrics
    """
    jepa.eval()

    # Storage for predictions and ground truth
    all_pred_rewards = []
    all_true_rewards = []
    all_pred_classes = []
    all_true_classes = []

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=num_batches, desc="Evaluating reward head")
        for i, batch in pbar:
            if i >= num_batches:
                break

            # Unpack batch (TrajectoryBatch format)
            if hasattr(batch, 'states'):
                # NamedTuple format
                obs = batch.states.to(device)  # [B, C, T, H, W]
                rewards = batch.metadata.get('rewards', None)
                if rewards is not None:
                    rewards = rewards.to(device)
            elif isinstance(batch, dict):
                obs = batch['obs'].to(device)
                rewards = batch.get('rewards', None)
                if rewards is not None:
                    rewards = rewards.to(device)
            else:
                # Tuple format
                obs, actions, rewards = batch[:3] if len(batch) >= 3 else (batch[0], None, None)
                obs = obs.to(device)
                if rewards is not None:
                    rewards = rewards.to(device)

            if rewards is None:
                print("Warning: No rewards in batch, skipping...")
                continue

            # Get batch size and sequence length
            B, C, T, H, W = obs.shape

            # Encode observations to get latent states
            # Encoder expects [B, C, T, H, W] and returns [B, D, T, H', W']
            latents = jepa.encoder(obs)  # [B, D, T, H', W']

            # Get latent dimensions
            D = latents.shape[1]
            H_prime, W_prime = latents.shape[3], latents.shape[4]

            # Predict reward logits for each timestep
            # Encoder returns [B, D, T, H', W'], permute for reward head
            logits = jepa.reward_head(latents)  # [B, T, 2]

            # Get probabilities for reward class (index 1)
            probs = torch.softmax(logits, dim=-1)[:, :, 1]  # [B, T]

            # Collect predictions and ground truth
            all_pred_rewards.append(probs.cpu().numpy())
            all_true_rewards.append(rewards.cpu().numpy())

            # Convert to binary classes (reward vs no reward)
            pred_classes = (probs > 0.5).long()
            true_classes = (rewards > 0.5).long()

            all_pred_classes.append(pred_classes.cpu().numpy())
            all_true_classes.append(true_classes.cpu().numpy())

            # Update progress bar with running stats
            if i % 5 == 0:
                temp_pred = np.concatenate(all_pred_classes, axis=0).flatten()
                temp_true = np.concatenate(all_true_classes, axis=0).flatten()
                acc = np.mean(temp_pred == temp_true)
                pbar.set_postfix({'Accuracy': f'{acc:.4f}'})

    # Concatenate all results
    pred_probs = np.concatenate(all_pred_rewards, axis=0).flatten()  # Predicted probabilities
    true_labels = np.concatenate(all_true_rewards, axis=0).flatten()  # Ground truth (continuous)
    pred_classes = np.concatenate(all_pred_classes, axis=0).flatten()
    true_classes = np.concatenate(all_true_classes, axis=0).flatten()

    # Compute calibration metrics (how well probabilities match binary labels)
    # For perfect calibration: prob should be close to 1 when reward=1, close to 0 when reward=0
    true_binary = (true_labels > 0.5).astype(float)
    brier_score = np.mean((pred_probs - true_binary) ** 2)  # Like MSE but for probabilities

    # Average predicted probability for each class
    mean_prob_no_reward = np.mean(pred_probs[true_classes == 0]) if np.any(true_classes == 0) else 0.0
    mean_prob_reward = np.mean(pred_probs[true_classes == 1]) if np.any(true_classes == 1) else 0.0

    # Classification accuracy (reward vs no reward)
    accuracy = np.mean(pred_classes == true_classes)

    # Per-class metrics
    # True positives, false positives, false negatives, true negatives
    tp = np.sum((pred_classes == 1) & (true_classes == 1))
    fp = np.sum((pred_classes == 1) & (true_classes == 0))
    fn = np.sum((pred_classes == 0) & (true_classes == 1))
    tn = np.sum((pred_classes == 0) & (true_classes == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Reward distribution
    num_rewards = np.sum(true_classes == 1)
    num_no_rewards = np.sum(true_classes == 0)
    reward_ratio = num_rewards / len(true_classes)

    results = {
        'brier_score': brier_score,
        'mean_prob_no_reward': mean_prob_no_reward,
        'mean_prob_reward': mean_prob_reward,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_samples': len(pred_probs),
        'num_rewards': num_rewards,
        'num_no_rewards': num_no_rewards,
        'reward_ratio': reward_ratio,
        'pred_rewards': pred_probs,  # Now probabilities, not scalars
        'true_rewards': true_labels,
        'pred_classes': pred_classes,
        'true_classes': true_classes,
    }

    return results


def plot_results(results, output_dir):
    """Plot reward prediction results for binary classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. ROC Curve
    ax = axes[0, 0]
    from sklearn.metrics import roc_curve, roc_auc_score

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(results['true_classes'], results['pred_rewards'])
    roc_auc = roc_auc_score(results['true_classes'], results['pred_rewards'])

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve - Reward Classification', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    # 2. Distribution of predicted probabilities
    ax = axes[0, 1]
    ax.hist(results['pred_rewards'][results['true_classes'] == 0],
            bins=50, alpha=0.6, label='No reward (true)', density=True, color='blue')
    ax.hist(results['pred_rewards'][results['true_classes'] == 1],
            bins=50, alpha=0.6, label='Reward (true)', density=True, color='orange')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Classification threshold')
    ax.set_xlabel('Predicted Probability of Reward', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Probability Distribution by True Class', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])

    # 3. Confusion matrix (normalized)
    ax = axes[1, 0]
    tp = np.sum((results['pred_classes'] == 1) & (results['true_classes'] == 1))
    fp = np.sum((results['pred_classes'] == 1) & (results['true_classes'] == 0))
    fn = np.sum((results['pred_classes'] == 0) & (results['true_classes'] == 1))
    tn = np.sum((results['pred_classes'] == 0) & (results['true_classes'] == 0))

    confusion = np.array([[tn, fp], [fn, tp]])

    # Normalize by row (true class) for better visualization
    confusion_norm = confusion.astype(float) / confusion.sum(axis=1, keepdims=True)

    im = ax.imshow(confusion_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Reward', 'Reward'], fontsize=11)
    ax.set_yticklabels(['No Reward', 'Reward'], fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')

    # Add text annotations with counts and percentages
    for i in range(2):
        for j in range(2):
            count = confusion[i, j]
            percentage = confusion_norm[i, j]
            text = ax.text(j, i, f'{count:,}\n({percentage:.1%})',
                          ha="center", va="center",
                          color="white" if percentage > 0.5 else "black",
                          fontsize=11, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Fraction of true class')

    # 4. Metrics summary
    ax = axes[1, 1]
    ax.axis('off')

    # Compute ROC-AUC for summary
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(results['true_classes'], results['pred_rewards'])

    summary_text = f"""
    Binary Reward Classification Summary
    {'='*40}

    ROC Metrics:
    • ROC-AUC: {roc_auc:.4f}

    Calibration Metrics:
    • Brier Score: {results['brier_score']:.4f}
    • Mean prob (no reward): {results['mean_prob_no_reward']:.4f}
    • Mean prob (reward): {results['mean_prob_reward']:.4f}

    Classification (threshold=0.5):
    • Accuracy: {results['accuracy']:.2%}
    • Precision: {results['precision']:.2%}
    • Recall: {results['recall']:.2%}
    • F1 Score: {results['f1']:.4f}

    Dataset Distribution:
    • Total samples: {results['num_samples']:,}
    • Rewards: {results['num_rewards']:,} ({results['reward_ratio']:.2%})
    • No rewards: {results['num_no_rewards']:,} ({1-results['reward_ratio']:.2%})

    {'='*40}
    """

    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()

    output_path = output_dir / 'reward_head_evaluation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved plot: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate reward prediction head')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--num_batches', type=int, default=50,
                        help='Number of batches to evaluate')
    parser.add_argument('--output_dir', type=str, default='visualizations/reward_eval',
                        help='Output directory for plots')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("REWARD HEAD EVALUATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Num batches: {args.num_batches}")
    print(f"Output dir: {args.output_dir}")

    # Load model
    print("\nLoading model...")
    jepa, cfg, data_config = load_jepa_from_checkpoint(
        args.checkpoint, args.config, device='auto'
    )
    device = next(jepa.parameters()).device
    print(f"✅ Model loaded on {device}")

    # Check if reward head exists
    if not hasattr(jepa, 'reward_head') or jepa.reward_head is None:
        print("\n❌ Error: Model does not have a reward head!")
        print("Make sure the checkpoint was trained with reward prediction enabled.")
        return

    print(f"✅ Reward head found: {type(jepa.reward_head).__name__}")

    # Load data
    print("\nLoading validation data...")
    cfg_data_dict = vars(cfg.data)
    train_loader, val_loader, _ = init_data(cfg.data.env_name, cfg_data_dict)
    print(f"✅ Validation loader ready: {len(val_loader)} batches available")

    # Evaluate
    print("\n" + "="*70)
    print("EVALUATING...")
    print("="*70)
    results = evaluate_reward_head(jepa, val_loader, device, args.num_batches)

    # Print results
    print("\n" + "="*70)
    print("RESULTS (Binary Classification)")
    print("="*70)
    print(f"\nProbability Calibration:")
    print(f"  Brier Score: {results['brier_score']:.4f} (lower is better)")
    print(f"  Mean prob (no reward class): {results['mean_prob_no_reward']:.4f}")
    print(f"  Mean prob (reward class):    {results['mean_prob_reward']:.4f}")

    print(f"\nClassification Metrics (threshold=0.5):")
    print(f"  Accuracy:  {results['accuracy']:.2%}")
    print(f"  Precision: {results['precision']:.2%}")
    print(f"  Recall:    {results['recall']:.2%}")
    print(f"  F1 Score:  {results['f1']:.4f}")

    print(f"\nDataset Distribution:")
    print(f"  Total samples: {results['num_samples']:,}")
    print(f"  Rewards:       {results['num_rewards']:,} ({results['reward_ratio']:.2%})")
    print(f"  No rewards:    {results['num_no_rewards']:,} ({1-results['reward_ratio']:.2%})")

    # Plot results
    print("\n" + "="*70)
    print("PLOTTING...")
    print("="*70)
    plot_results(results, args.output_dir)

    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📊 View results in: {args.output_dir}")


if __name__ == '__main__':
    main()
