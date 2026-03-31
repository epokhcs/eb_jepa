"""Quick verification script for human gameplay dataset."""

import pickle
import numpy as np

# Load the dataset
print("Loading dataset...")
with open('data/systematic_sweep/breakout_human_gameplay.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"\n{'='*70}")
print("DATASET SUMMARY")
print(f"{'='*70}")

# Basic stats
print(f"Total trajectories: {len(data['trajectories'])}")
print(f"Metadata: {data['metadata']}")

# Per-trajectory stats
print(f"\n{'='*70}")
print("TRAJECTORY DETAILS")
print(f"{'='*70}")

total_frames = 0
total_rewards = 0
valid_reward_count = 0

for i, traj in enumerate(data['trajectories']):
    frames = traj['length']
    rewards_sum = np.nansum(traj['rewards'])  # Use nansum to handle NaN
    rewards_valid = np.sum(~np.isnan(traj['rewards']))

    total_frames += frames
    total_rewards += rewards_sum
    valid_reward_count += rewards_valid

    print(f"Trajectory {i}:")
    print(f"  Trial ID: {traj.get('trial_id', 'N/A')}")
    print(f"  Skill level: {traj.get('skill_level', 'N/A')}")
    print(f"  Frames: {frames}")
    print(f"  Observations shape: {traj['observations'].shape}")
    print(f"  Actions shape: {traj['actions'].shape}")
    print(f"  Rewards shape: {traj['rewards'].shape}")
    print(f"  Total reward: {rewards_sum:.0f}")
    print(f"  Valid reward values: {rewards_valid}/{frames}")
    print(f"  NaN rewards: {frames - rewards_valid}")

    # Action distribution for this trajectory
    unique_actions, counts = np.unique(traj['actions'][~np.isnan(traj['actions'])], return_counts=True)
    print(f"  Actions: {dict(zip(unique_actions.astype(int), counts))}")
    print()

print(f"{'='*70}")
print("OVERALL STATS")
print(f"{'='*70}")
print(f"Total frames: {total_frames:,}")
print(f"Total rewards: {total_rewards:.0f}")
print(f"Valid reward values: {valid_reward_count:,}/{total_frames:,}")
print(f"NaN rewards: {total_frames - valid_reward_count:,}")

# Overall action distribution
all_actions = np.concatenate([t['actions'] for t in data['trajectories']])
all_actions = all_actions[~np.isnan(all_actions)]
unique_actions, counts = np.unique(all_actions, return_counts=True)
action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']

print(f"\nAction distribution:")
for action, count in zip(unique_actions.astype(int), counts):
    pct = count / len(all_actions) * 100
    if action < len(action_names):
        print(f"  {action_names[action]}: {pct:.1f}% ({count:,} frames)")

print(f"\n✅ Dataset verification complete!")
