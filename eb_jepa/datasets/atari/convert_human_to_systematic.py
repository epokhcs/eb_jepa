"""
Convert human gameplay data from ATARI-HEAD dataset to systematic_sweep format.

This script processes selected Breakout trials and converts them into a pickle file
compatible with the SystematicAtariDataset loader.

Usage:
    python -m eb_jepa.datasets.atari.convert_human_to_systematic
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from eb_jepa.datasets.atari.human_loader import (
    extract_breakout_zip,
    extract_human_trial,
    convert_to_trajectory_format
)


def main():
    """Main conversion function."""

    # Configuration
    SELECTED_TRIALS = {
        'high': [205, 218],
        'medium': [307, 472],
        'low': [58, 153]
    }

    OUTPUT_PATH = "data/systematic_sweep/breakout_human_gameplay.pkl"
    ZIP_PATH = "data/human/breakout.zip"
    METADATA_PATH = "data/human/meta_data.csv"

    print("=" * 70)
    print("CONVERTING HUMAN GAMEPLAY DATA TO SYSTEMATIC SWEEP FORMAT")
    print("=" * 70)

    # 1. Extract breakout.zip if not already extracted
    base_path = extract_breakout_zip(ZIP_PATH, "data/human/")

    # 2. Load metadata.csv for reference
    print(f"\nLoading metadata from {METADATA_PATH}...")
    metadata = pd.read_csv(METADATA_PATH)
    print(f"✅ Loaded {len(metadata)} trials from metadata")

    # Show selected trials info
    print("\nSelected trials:")
    for category, trial_ids in SELECTED_TRIALS.items():
        print(f"\n{category.upper()} SCORE:")
        for trial_id in trial_ids:
            trial_info = metadata[metadata['trial_id'] == trial_id]
            if not trial_info.empty:
                score = trial_info.iloc[0]['highest_score']
                frames = trial_info.iloc[0]['total_frame']
                subject = trial_info.iloc[0]['subject_id']
                print(f"  Trial {trial_id}: Score {score}, {frames} frames, Subject {subject}")

    # 3. Process each trial
    print("\n" + "=" * 70)
    print("PROCESSING TRIALS")
    print("=" * 70)

    all_trajectories = []
    for category, trial_ids in SELECTED_TRIALS.items():
        for trial_id in trial_ids:
            try:
                # Extract trial data
                trial_data = extract_human_trial(
                    trial_id,
                    base_path=base_path,
                    metadata_df=metadata
                )

                # Convert to trajectory format
                trajectories = convert_to_trajectory_format(trial_data)

                # Add category metadata
                for traj in trajectories:
                    traj['skill_level'] = category  # high/medium/low
                    traj['trial_id'] = trial_id
                    traj['subject'] = trial_data['subject']

                all_trajectories.extend(trajectories)

                print(f"  ✅ Added {len(trajectories)} trajectories from trial {trial_id}")

            except Exception as e:
                print(f"  ❌ Error processing trial {trial_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 4. Create dataset structure matching systematic_sweep
    print("\n" + "=" * 70)
    print("CREATING DATASET")
    print("=" * 70)

    total_frames = sum(t['length'] for t in all_trajectories)
    total_rewards = sum(t['rewards'].sum() for t in all_trajectories)

    dataset = {
        'trajectories': all_trajectories,
        'paddle_position_map': list(range(len(all_trajectories))),
        'metadata': {
            'game': 'Breakout',
            'paddle_positions': len(all_trajectories),  # Variable
            'episodes_per_position': 1,  # Variable per trial
            'trajectory_length': 'variable',  # Episodes vary in length
            'total_trajectories': len(all_trajectories),
            'paddle_policy': 'human',
            'source': 'ATARI-HEAD',
            'skill_levels': ['high', 'medium', 'low'],
            'trials': SELECTED_TRIALS,
            'total_frames': total_frames,
            'total_rewards': float(total_rewards),
        }
    }

    # 5. Create output directory if needed
    output_dir = Path(OUTPUT_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Save to pickle
    print(f"\nSaving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    # 7. Summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"✅ Saved {len(all_trajectories)} trajectories to {OUTPUT_PATH}")
    print(f"📊 Total frames: {total_frames:,}")
    print(f"🏆 Total rewards: {total_rewards:.0f}")
    print(f"📈 Average reward per frame: {total_rewards / total_frames:.4f}")

    # Action distribution
    print("\nAction distribution:")
    all_actions = np.concatenate([t['actions'] for t in all_trajectories])
    unique, counts = np.unique(all_actions, return_counts=True)
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    for action, count in zip(unique, counts):
        pct = count / len(all_actions) * 100
        print(f"  {action_names[int(action)]}: {pct:.1f}% ({count:,} frames)")

    # Reward distribution
    print("\nReward distribution:")
    all_rewards = np.concatenate([t['rewards'] for t in all_trajectories])
    unique_rewards, reward_counts = np.unique(all_rewards, return_counts=True)
    for reward, count in zip(unique_rewards, reward_counts):
        pct = count / len(all_rewards) * 100
        print(f"  Reward {reward:.0f}: {pct:.1f}% ({count:,} frames)")

    # Trajectory lengths
    print("\nTrajectory statistics:")
    lengths = [t['length'] for t in all_trajectories]
    print(f"  Number of trajectories: {len(lengths)}")
    print(f"  Min length: {min(lengths):,} frames")
    print(f"  Max length: {max(lengths):,} frames")
    print(f"  Mean length: {np.mean(lengths):.0f} frames")
    print(f"  Median length: {np.median(lengths):.0f} frames")

    print("\n✅ Dataset ready for training!")
    print(f"Load with: SystematicAtariDataset(data_path='{OUTPUT_PATH}')")


if __name__ == "__main__":
    main()
