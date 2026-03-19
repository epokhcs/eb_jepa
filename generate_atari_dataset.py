#!/usr/bin/env python3
"""
Script to pre-generate ATARI trajectory dataset.

This collects trajectories using a random policy and saves them to disk
for faster training (avoids regenerating data each epoch).
"""

import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, '/Users/pdiprodi/DevOps/github/eb_jepa')

from eb_jepa.datasets.atari import AtariEnv, AtariConfig


def collect_trajectory(env, num_steps=200):
    """Collect a single trajectory."""
    states = []
    actions = []
    rewards = []
    dones = []

    obs, info = env.reset()
    states.append(obs)

    for step in range(num_steps):
        # Random action
        action = env.env.action_space.sample()
        actions.append(action)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        rewards.append(reward)
        done = terminated or truncated
        dones.append(done)

        # Reset if done
        if done and step < num_steps - 1:
            obs, info = env.reset()
            states[-1] = obs

    # Remove last state
    states = states[:-1]

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
    }


def generate_dataset(
    game_name='Breakout',
    num_trajectories=10000,
    num_steps=200,
    output_dir='data/atari',
    split='train'
):
    """Generate and save ATARI dataset."""

    print(f"Generating {num_trajectories} {game_name} trajectories...")
    print(f"Steps per trajectory: {num_steps}")

    # Create config
    config = AtariConfig(
        game_name=game_name,
        device='cpu',
        n_steps=num_steps,
    )

    # Create environment
    env = AtariEnv(config)
    print(f"Environment: {game_name}")
    print(f"Action space: {env.get_action_space_info()}")
    print(f"Observation space: {env.get_observation_space_info()}")

    # Collect trajectories
    trajectories = []
    for i in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        traj = collect_trajectory(env, num_steps)
        trajectories.append(traj)

    env.close()

    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{game_name.lower()}_{split}_{num_trajectories}.pkl"

    print(f"\nSaving dataset to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(trajectories, f)

    # Print statistics
    total_steps = sum(len(t['actions']) for t in trajectories)
    total_rewards = sum(sum(t['rewards']) for t in trajectories)
    avg_reward = total_rewards / num_trajectories

    print(f"\nDataset Statistics:")
    print(f"  Total trajectories: {num_trajectories}")
    print(f"  Total steps: {total_steps}")
    print(f"  Average reward per trajectory: {avg_reward:.2f}")
    print(f"  Saved to: {output_file}")

    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ATARI pretraining dataset')
    parser.add_argument('--game', default='Breakout', help='ATARI game name')
    parser.add_argument('--num_trajectories', type=int, default=10000,
                       help='Number of trajectories to collect')
    parser.add_argument('--num_steps', type=int, default=200,
                       help='Steps per trajectory')
    parser.add_argument('--output_dir', default='data/atari',
                       help='Output directory')
    parser.add_argument('--split', default='train', choices=['train', 'val'],
                       help='Dataset split')

    args = parser.parse_args()

    generate_dataset(
        game_name=args.game,
        num_trajectories=args.num_trajectories,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        split=args.split,
    )
