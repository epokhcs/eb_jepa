#!/usr/bin/env python3
"""
Compare planning with latent variance vs predicted reward objectives.

This script runs both objectives and generates a comparative analysis.

Usage:
    python -m examples.ac_video_jepa.compare_planning_objectives \
        --checkpoint path/to/checkpoint.pth.tar \
        --planner mppi \
        --num_episodes 20
"""

import argparse
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.ac_video_jepa.test_planning_with_rewards import test_planning
import matplotlib.pyplot as plt
import numpy as np


def compare_objectives(checkpoint_path, planner_type="mppi", num_episodes=20, horizon=16):
    """Run planning with both objectives and compare results."""

    print("\n" + "="*80)
    print("COMPARATIVE PLANNING EVALUATION")
    print("Latent Variance (baseline) vs Predicted Reward (new)")
    print("="*80)

    # Test 1: Latent variance (baseline)
    print("\n\n" + "#"*80)
    print("# TEST 1: LATENT VARIANCE OBJECTIVE (BASELINE)")
    print("#"*80 + "\n")

    results_variance = test_planning(
        checkpoint_path=checkpoint_path,
        planner_type=planner_type,
        objective="latent_variance",
        num_episodes=num_episodes,
        horizon=horizon,
    )

    # Test 2: Predicted reward
    print("\n\n" + "#"*80)
    print("# TEST 2: PREDICTED REWARD OBJECTIVE (NEW)")
    print("#"*80 + "\n")

    results_reward = test_planning(
        checkpoint_path=checkpoint_path,
        planner_type=planner_type,
        objective="predicted_reward",
        num_episodes=num_episodes,
        horizon=horizon,
    )

    # Compare results
    print("\n\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)

    print(f"\n📊 Mean Reward Comparison:")
    print(f"  Latent Variance:   {results_variance['mean_reward']:.2f} ± {results_variance['std_reward']:.2f}")
    print(f"  Predicted Reward:  {results_reward['mean_reward']:.2f} ± {results_reward['std_reward']:.2f}")

    improvement = results_reward['mean_reward'] - results_variance['mean_reward']
    pct_improvement = (improvement / max(results_variance['mean_reward'], 0.1)) * 100

    print(f"\n📈 Improvement:")
    print(f"  Absolute: {improvement:+.2f} points")
    print(f"  Relative: {pct_improvement:+.1f}%")

    if pct_improvement > 200:
        print(f"  🎉 EXCELLENT! More than 2x better with reward prediction!")
    elif pct_improvement > 100:
        print(f"  ✅ GREAT! More than 2x better with reward prediction!")
    elif pct_improvement > 50:
        print(f"  ✅ GOOD! Significant improvement with reward prediction!")
    elif pct_improvement > 0:
        print(f"  ⚠️  Some improvement, but less than expected.")
    else:
        print(f"  ❌ Reward prediction not helping. Check training!")

    print(f"\n🎮 Action Distribution Comparison:")
    action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
    print(f"\n  {'Action':<10} {'Variance %':>12} {'Reward %':>12} {'Change':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for i, name in enumerate(action_names):
        var_pct = (results_variance['action_counts'][i] / sum(results_variance['action_counts'].values())) * 100
        rew_pct = (results_reward['action_counts'][i] / sum(results_reward['action_counts'].values())) * 100
        change = rew_pct - var_pct
        print(f"  {name:<10} {var_pct:>11.1f}% {rew_pct:>11.1f}% {change:>+11.1f}%")

    # Save comparison plot
    results_dir = Path(checkpoint_path).parent / "planning_results"
    results_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean reward comparison
    ax = axes[0]
    objectives = ['Latent\nVariance', 'Predicted\nReward']
    means = [results_variance['mean_reward'], results_reward['mean_reward']]
    stds = [results_variance['std_reward'], results_reward['std_reward']]

    bars = ax.bar(objectives, means, yerr=stds, capsize=10,
                  color=['#3498db', '#2ecc71'], alpha=0.7)
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title(f'{planner_type.upper()} Planner: Objective Comparison\n({num_episodes} episodes)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
               f'{mean:.1f}±{std:.1f}',
               ha='center', va='bottom', fontweight='bold')

    # Action distribution comparison
    ax = axes[1]
    x = np.arange(len(action_names))
    width = 0.35

    var_pcts = [results_variance['action_counts'][i] / sum(results_variance['action_counts'].values()) * 100
                for i in range(4)]
    rew_pcts = [results_reward['action_counts'][i] / sum(results_reward['action_counts'].values()) * 100
                for i in range(4)]

    ax.bar(x - width/2, var_pcts, width, label='Latent Variance',
           color='#3498db', alpha=0.7)
    ax.bar(x + width/2, rew_pcts, width, label='Predicted Reward',
           color='#2ecc71', alpha=0.7)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Action Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    comparison_file = results_dir / f"{planner_type}_comparison.png"
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to {comparison_file}")

    plt.show()

    # Save comparison summary
    summary_file = results_dir / f"{planner_type}_comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PLANNING OBJECTIVE COMPARISON\n")
        f.write("="*80 + "\n\n")

        f.write(f"Planner: {planner_type.upper()}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Horizon: {horizon}\n\n")

        f.write("RESULTS:\n")
        f.write(f"  Latent Variance:  {results_variance['mean_reward']:.2f} ± {results_variance['std_reward']:.2f}\n")
        f.write(f"  Predicted Reward: {results_reward['mean_reward']:.2f} ± {results_reward['std_reward']:.2f}\n\n")

        f.write("IMPROVEMENT:\n")
        f.write(f"  Absolute: {improvement:+.2f} points\n")
        f.write(f"  Relative: {pct_improvement:+.1f}%\n\n")

        f.write("INTERPRETATION:\n")
        if pct_improvement > 200:
            f.write("  🎉 EXCELLENT! Reward prediction working very well!\n")
        elif pct_improvement > 100:
            f.write("  ✅ GREAT! Significant improvement from reward prediction!\n")
        elif pct_improvement > 50:
            f.write("  ✅ GOOD! Meaningful improvement from reward prediction!\n")
        elif pct_improvement > 0:
            f.write("  ⚠️  Modest improvement. Consider longer training.\n")
        else:
            f.write("  ❌ No improvement. Check reward head training!\n")

    print(f"✅ Summary saved to {summary_file}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    return {
        'latent_variance': results_variance,
        'predicted_reward': results_reward,
        'improvement_abs': improvement,
        'improvement_pct': pct_improvement,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare planning objectives')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--planner', type=str, default='mppi', choices=['mppi', 'cem'],
                       help='Planner to use (mppi or cem)')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes per objective')
    parser.add_argument('--horizon', type=int, default=16,
                       help='Planning horizon')

    args = parser.parse_args()

    compare_objectives(
        checkpoint_path=args.checkpoint,
        planner_type=args.planner,
        num_episodes=args.num_episodes,
        horizon=args.horizon,
    )


if __name__ == '__main__':
    main()
