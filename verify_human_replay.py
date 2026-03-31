"""
Verify human gameplay data by replaying in ATARI simulator.

This script loads a human gameplay trajectory and replays the actions
in the actual ATARI environment to verify:
1. Actions are correctly formatted
2. Reward sequences match (or are similar)
3. The data will work for planning later
"""

import pickle
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eb_jepa.datasets.atari.config import AtariConfig
from eb_jepa.datasets.atari.env import AtariEnv


def verify_trajectory_replay(trajectory_idx=0, max_steps=500):
    """
    Replay a human trajectory in the simulator and compare rewards.

    Args:
        trajectory_idx: Which trajectory to test (0-5 for our 6 trials)
        max_steps: Maximum steps to replay (for quick testing)
    """

    # Load human gameplay dataset
    print("Loading human gameplay dataset...")
    with open('data/systematic_sweep/breakout_human_gameplay.pkl', 'rb') as f:
        data = pickle.load(f)

    if trajectory_idx >= len(data['trajectories']):
        print(f"Error: Only {len(data['trajectories'])} trajectories available")
        return

    traj = data['trajectories'][trajectory_idx]

    print(f"\n{'='*70}")
    print(f"REPLAYING TRAJECTORY {trajectory_idx}")
    print(f"{'='*70}")
    print(f"Trial ID: {traj.get('trial_id', 'N/A')}")
    print(f"Skill level: {traj.get('skill_level', 'N/A')}")
    print(f"Total frames: {traj['length']}")
    print(f"Recorded total reward: {np.nansum(traj['rewards']):.0f}")
    print()

    # Limit steps for quick verification
    test_steps = min(max_steps, traj['length'])
    human_actions = traj['actions'][:test_steps]
    human_rewards = traj['rewards'][:test_steps]

    # Create ATARI environment
    print("Creating ATARI environment...")
    config = AtariConfig(
        game_name="Breakout",
        batch_size=1,
    )
    env = AtariEnv(config, render_mode=None)  # No rendering for speed

    # Reset environment
    obs, info = env.reset()
    print(f"Environment ready. Replaying {test_steps} steps...")
    print()

    # Replay human actions and collect rewards
    simulator_rewards = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # NOOP, FIRE, RIGHT, LEFT

    for step in range(test_steps):
        human_action = int(human_actions[step])
        human_reward = human_rewards[step]

        # Execute action in simulator
        obs, sim_reward, terminated, truncated, info = env.step(human_action)
        simulator_rewards.append(sim_reward)
        action_counts[human_action] += 1

        # Print first few steps for debugging
        if step < 10:
            action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            print(f"Step {step}: Action={action_names[human_action]}, "
                  f"Human reward={human_reward:.0f}, Sim reward={sim_reward:.0f}")

        # Stop if episode ends in simulator
        if terminated or truncated:
            print(f"\nEpisode terminated in simulator at step {step}")
            # Trim human data to match
            human_rewards = human_rewards[:step+1]
            break

    env.close()

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")

    # Handle NaN values in human rewards
    human_rewards_valid = human_rewards[~np.isnan(human_rewards)]
    human_total = np.sum(human_rewards_valid)
    sim_total = np.sum(simulator_rewards)

    print(f"Steps replayed: {len(simulator_rewards)}")
    print(f"\nAction distribution:")
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    for action, count in action_counts.items():
        pct = count / len(simulator_rewards) * 100
        print(f"  {action_names[action]}: {count} ({pct:.1f}%)")

    print(f"\nReward comparison:")
    print(f"  Human recorded total: {human_total:.0f}")
    print(f"  Simulator total: {sim_total:.0f}")
    print(f"  Difference: {abs(sim_total - human_total):.0f}")

    # Count matching rewards (where human data is valid)
    valid_mask = ~np.isnan(human_rewards[:len(simulator_rewards)])
    matching_rewards = np.sum(
        human_rewards[:len(simulator_rewards)][valid_mask] ==
        np.array(simulator_rewards)[valid_mask]
    )
    total_valid = np.sum(valid_mask)

    print(f"\nFrame-by-frame reward match:")
    print(f"  Matching: {matching_rewards}/{total_valid} ({matching_rewards/total_valid*100:.1f}%)")
    print(f"  Non-matching: {total_valid - matching_rewards}")

    # Show reward sequence comparison (first 50 non-zero rewards)
    print(f"\nFirst non-zero rewards (Human vs Sim):")
    nonzero_count = 0
    for i in range(min(len(simulator_rewards), len(human_rewards))):
        if not np.isnan(human_rewards[i]) and (human_rewards[i] != 0 or simulator_rewards[i] != 0):
            print(f"  Step {i}: Human={human_rewards[i]:.0f}, Sim={simulator_rewards[i]:.0f}")
            nonzero_count += 1
            if nonzero_count >= 20:
                break

    # Overall verdict
    print(f"\n{'='*70}")
    reward_match_pct = matching_rewards / total_valid * 100 if total_valid > 0 else 0
    if reward_match_pct > 90:
        print("✅ VERIFICATION PASSED: Rewards match well (>90%)")
        print("   Data should work correctly for training and planning.")
    elif reward_match_pct > 70:
        print("⚠️  VERIFICATION WARNING: Rewards mostly match but some differences")
        print("   This may be acceptable - ATARI can have non-determinism.")
    else:
        print("❌ VERIFICATION FAILED: Rewards don't match")
        print("   Possible issues: action mapping, environment setup, or data corruption.")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=int, default=0,
                       help='Which trajectory to replay (0-5)')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum steps to replay for quick testing')
    args = parser.parse_args()

    verify_trajectory_replay(args.trajectory, args.max_steps)
