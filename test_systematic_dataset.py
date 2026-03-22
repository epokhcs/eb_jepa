#!/usr/bin/env python3
"""
Test loading and visualizing systematic paddle sweep dataset.
"""

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from eb_jepa.datasets.atari.systematic_dataset import SystematicAtariDataset
from eb_jepa.datasets.atari.config import AtariConfig


def test_systematic_dataset():
    """Test loading systematic dataset."""
    print("=" * 70)
    print("TESTING SYSTEMATIC DATASET LOADING")
    print("=" * 70)

    # Create config
    config = AtariConfig(
        game_name="Breakout",
        batch_size=4,
        sample_length=17,
        normalize=True,
    )

    # Load dataset
    dataset_path = "data/systematic_sweep/breakout_systematic_paddle_sweep.pkl"
    dataset = SystematicAtariDataset(config, dataset_path)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Game: {dataset.metadata['game']}")
    print(f"Paddle positions: {dataset.metadata['paddle_positions']}")

    # Get a sample
    print("\n" + "=" * 70)
    print("LOADING SAMPLE TRAJECTORY")
    print("=" * 70)

    batch = dataset[0]
    print(f"\nStates shape: {batch.states.shape}")  # [C, T, H, W]
    print(f"Actions shape: {batch.actions.shape}")  # [1, T]
    print(f"Rewards shape: {batch.metadata['rewards'].shape}")  # [T]
    print(f"Paddle start position: {batch.metadata['paddle_start_position']}")
    print(f"Position index: {batch.metadata['position_index']}")

    # Verify data range
    print(f"\nStates min: {batch.states.min():.4f}, max: {batch.states.max():.4f}")
    print(f"Actions unique values: {torch.unique(batch.actions)}")
    print(f"Total reward: {batch.metadata['rewards'].sum():.2f}")

    # Visualize first few frames
    print("\n" + "=" * 70)
    print("VISUALIZING FIRST 4 FRAMES")
    print("=" * 70)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        frame = batch.states[0, i].cpu().numpy()  # [H, W]
        axes[i].imshow(frame, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f"Frame {i}\nAction: {int(batch.actions[0, i])}")
        axes[i].axis('off')

    plt.tight_layout()
    output_path = "visualizations/systematic_dataset_sample.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    # Test paddle position distribution
    print("\n" + "=" * 70)
    print("PADDLE POSITION DISTRIBUTION")
    print("=" * 70)

    position_counts = {}
    for idx in range(len(dataset)):
        batch = dataset[idx]
        pos_idx = batch.metadata['position_index']
        position_counts[pos_idx] = position_counts.get(pos_idx, 0) + 1

    print(f"\nPosition distribution:")
    for pos in sorted(position_counts.keys())[:10]:
        print(f"  Position {pos}: {position_counts[pos]} trajectories")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_systematic_dataset()
