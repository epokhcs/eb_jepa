#!/usr/bin/env python3
"""
Generate ATARI dataset with systematic paddle position sweeping.

This script uses OCAtari to systematically vary paddle positions before firing,
ensuring the world model learns from diverse initial conditions.

Usage:
    uv run python -m eb_jepa.datasets.atari.generate_systematic_sweep \
        --game Breakout \
        --paddle_positions 20 \
        --episodes_per_position 3 \
        --output_dir data/systematic_sweep \
        --paddle_policy tracking

Strategy:
1. Reset game
2. Move paddle all the way to the left
3. Fire ball from leftmost position
4. Let episode play out (with selected paddle policy)
5. Reset, move one step right from left
6. Fire again
7. Repeat until we've covered the full paddle range

Paddle Policies:
- static: Paddle stays still (NOOP only) - learns passive ball physics
- random: Paddle moves randomly - maximum action diversity
- tracking: Paddle tracks ball position - realistic gameplay
- mixed: 70% tracking + 30% random - balanced approach (recommended)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from ocatari.core import OCAtari
except ImportError:
    print("ERROR: OCAtari not installed. Install with: uv pip install ocatari")
    import sys
    sys.exit(1)


class PaddlePolicy:
    """Base class for paddle movement policies during trajectory collection."""

    def __init__(self, env):
        self.env = env
        self.NOOP = 0
        self.FIRE = 1
        self.RIGHT = 2
        self.LEFT = 3

    def get_action(self, step_count):
        """Return action based on current state. Override in subclasses."""
        raise NotImplementedError

    def get_paddle_position(self):
        """Get current paddle x position from OCAtari objects."""
        for obj in self.env.objects:
            if obj.category.lower() == "player":
                return obj.x
        return None

    def get_ball_position(self):
        """Get current ball x position from OCAtari objects."""
        for obj in self.env.objects:
            if obj.category.lower() == "ball":
                return obj.x
        return None


class StaticPaddlePolicy(PaddlePolicy):
    """Paddle stays static (NOOP only). Good for learning passive ball physics."""

    def get_action(self, step_count):
        return self.NOOP


class RandomPaddlePolicy(PaddlePolicy):
    """Paddle moves randomly. Maximum action diversity."""

    def __init__(self, env, p_move=0.3):
        super().__init__(env)
        self.p_move = p_move

    def get_action(self, step_count):
        if np.random.random() < self.p_move:
            return np.random.choice([self.LEFT, self.RIGHT])
        else:
            return self.NOOP


class BallTrackingPolicy(PaddlePolicy):
    """Paddle tracks ball position. Realistic gameplay."""

    def get_action(self, step_count):
        paddle_x = self.get_paddle_position()
        ball_x = self.get_ball_position()

        if paddle_x is None or ball_x is None:
            return self.NOOP

        # Move paddle toward ball with small tolerance
        tolerance = 5
        if ball_x < paddle_x - tolerance:
            return self.LEFT
        elif ball_x > paddle_x + tolerance:
            return self.RIGHT
        else:
            return self.NOOP


class MixedPolicy(PaddlePolicy):
    """70% ball tracking + 30% random. Balanced approach (recommended)."""

    def __init__(self, env, noise_prob=0.3):
        super().__init__(env)
        self.tracking_policy = BallTrackingPolicy(env)
        self.random_policy = RandomPaddlePolicy(env, p_move=0.5)
        self.noise_prob = noise_prob

    def get_action(self, step_count):
        if np.random.random() < self.noise_prob:
            return self.random_policy.get_action(step_count)
        else:
            return self.tracking_policy.get_action(step_count)


def create_paddle_policy(policy_name, env):
    """Factory function to create paddle policy from name."""
    policies = {
        "static": StaticPaddlePolicy,
        "random": RandomPaddlePolicy,
        "tracking": BallTrackingPolicy,
        "mixed": MixedPolicy,
    }

    if policy_name not in policies:
        raise ValueError(f"Unknown policy: {policy_name}. Choose from: {list(policies.keys())}")

    return policies[policy_name](env)


class SystematicPaddleDataGenerator:
    """Generate training data with systematic paddle position sweeping."""

    def __init__(self, game_name="Breakout", mode="ram"):
        self.game_name = game_name
        self.env = OCAtari(game_name, mode=mode, hud=False, render_mode='rgb_array')

        # Action mapping for Breakout
        self.NOOP = 0
        self.FIRE = 1
        self.RIGHT = 2
        self.LEFT = 3

        # Initialize heatmaps for paddle and ball tracking
        # Breakout screen is 210x160, we'll track at full resolution
        self.screen_height = 210
        self.screen_width = 160
        self.paddle_heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.int32)
        self.ball_heatmap = np.zeros((self.screen_height, self.screen_width), dtype=np.int32)

        print(f"Initialized OCAtari environment: {game_name}")
        print(f"Action space: {self.env.action_space.n} actions")
        print(f"Heatmap tracking enabled: {self.screen_width}x{self.screen_height}")

    def get_paddle_position(self):
        """Get current paddle x position from OCAtari objects."""
        for obj in self.env.objects:
            if obj.category.lower() == "player":
                return obj.x
        return None

    def get_paddle_range(self):
        """Determine the full range of paddle positions by moving it fully left and right."""
        print("\nDetermining paddle movement range...")

        # Move paddle all the way to the left
        obs, info = self.env.reset()
        for _ in range(100):
            self.env.step(self.LEFT)
        left_pos = self.get_paddle_position()

        # Move paddle all the way to the right
        for _ in range(200):
            self.env.step(self.RIGHT)
        right_pos = self.get_paddle_position()

        print(f"Paddle range: {left_pos} (left) to {right_pos} (right)")
        print(f"Total range: {right_pos - left_pos} pixels")

        return left_pos, right_pos

    def get_paddle_step_size(self):
        """Determine how many pixels the paddle moves per RIGHT action."""
        obs, info = self.env.reset()

        # Move to left
        for _ in range(100):
            self.env.step(self.LEFT)
        pos1 = self.get_paddle_position()

        # Move right once
        self.env.step(self.RIGHT)
        pos2 = self.get_paddle_position()

        step_size = pos2 - pos1
        print(f"Paddle step size: {step_size} pixels per RIGHT action")

        return step_size

    def update_heatmaps(self):
        """Update heatmaps with current paddle and ball positions."""
        for obj in self.env.objects:
            category = obj.category.lower()

            if category == "player":  # Paddle
                # Get object bounds (x, y, w, h)
                x, y, w, h = obj.x, obj.y, obj.w, obj.h
                # Increment heatmap for all pixels in the object's bounding box
                y_start = max(0, y)
                y_end = min(self.screen_height, y + h)
                x_start = max(0, x)
                x_end = min(self.screen_width, x + w)
                self.paddle_heatmap[y_start:y_end, x_start:x_end] += 1

            elif category == "ball":  # Ball
                # Get object bounds (x, y, w, h)
                x, y, w, h = obj.x, obj.y, obj.w, obj.h
                # Increment heatmap for all pixels in the object's bounding box
                y_start = max(0, y)
                y_end = min(self.screen_height, y + h)
                x_start = max(0, x)
                x_end = min(self.screen_width, x + w)
                self.ball_heatmap[y_start:y_end, x_start:x_end] += 1

    def move_paddle_to_left(self, max_steps=50):
        """Move paddle all the way to the left."""
        for _ in range(max_steps):
            # Don't track heatmaps during positioning - just moving to start position
            obs, reward, terminated, truncated, info = self.env.step(self.LEFT)
            if terminated or truncated:
                return None  # Episode ended unexpectedly
        return self.get_paddle_position()

    def move_paddle_right_n_steps(self, n_steps):
        """Move paddle n steps to the right from current position."""
        for _ in range(n_steps):
            # Don't track heatmaps during positioning - just moving to firing position
            obs, reward, terminated, truncated, info = self.env.step(self.RIGHT)
            if terminated or truncated:
                return None  # Episode ended unexpectedly
        return self.get_paddle_position()

    def fire_ball(self):
        """Fire the ball."""
        # Update heatmaps before firing
        self.update_heatmaps()
        obs, reward, terminated, truncated, info = self.env.step(self.FIRE)
        return obs, reward, terminated, truncated

    def collect_trajectory(self, max_length=200, paddle_policy=None):
        """Collect trajectory after firing until episode ends or max_length reached.

        Args:
            max_length: Maximum number of frames to collect
            paddle_policy: PaddlePolicy instance for controlling paddle during collection
        """
        observations = []
        actions = []
        rewards = []

        step_count = 0
        terminated = False
        truncated = False

        # Use static policy if none provided (backward compatibility)
        if paddle_policy is None:
            paddle_policy = StaticPaddlePolicy(self.env)

        while not terminated and not truncated and step_count < max_length:
            # Update heatmaps with current object positions
            self.update_heatmaps()

            # Get action from policy
            action = paddle_policy.get_action(step_count)

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)

            step_count += 1

        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'length': step_count,
        }

    def generate_systematic_dataset(
        self,
        output_dir,
        paddle_positions=None,  # If None, auto-detect full range
        episodes_per_position=3,  # Episodes to collect at each position
        trajectory_length=200,
        paddle_policy="static",  # Policy for paddle movement during collection
    ):
        """Generate dataset with systematic paddle sweeping.

        Args:
            output_dir: Directory to save dataset
            paddle_positions: Number of positions to sample (None = auto-detect)
            episodes_per_position: Episodes per paddle position
            trajectory_length: Max frames per trajectory
            paddle_policy: Paddle movement policy during collection
                - "static": Paddle stays still (NOOP) - default
                - "random": Random paddle movements
                - "tracking": Paddle tracks ball position (recommended for JEPA)
                - "mixed": 70% tracking + 30% random (recommended for JEPA)
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create paddle policy instance
        policy = create_paddle_policy(paddle_policy, self.env)
        print(f"Using paddle policy: {paddle_policy} ({type(policy).__name__})")

        # Auto-detect paddle range if not specified
        if paddle_positions is None:
            left_pos, right_pos = self.get_paddle_range()
            step_size = self.get_paddle_step_size()
            paddle_positions = int((right_pos - left_pos) / step_size) + 1
            print(f"Auto-detected paddle positions: {paddle_positions}")
        else:
            # Manual mode: still need to know the range
            left_pos, right_pos = self.get_paddle_range()
            step_size = self.get_paddle_step_size()
            print(f"Using manual paddle positions: {paddle_positions}")
            print(f"Note: This will sample {paddle_positions} evenly-spaced positions")
            print(f"      from x={left_pos} to x={right_pos} (step={step_size})")

        print("\n" + "="*70)
        print("SYSTEMATIC PADDLE SWEEP DATA GENERATION")
        print("="*70)
        print(f"Target paddle positions: {paddle_positions}")
        print(f"Episodes per position: {episodes_per_position}")
        print(f"Total episodes: {paddle_positions * episodes_per_position}")
        print(f"Trajectory length: {trajectory_length} steps")
        print("="*70 + "\n")

        all_trajectories = []
        paddle_position_map = []  # Track which paddle position each trajectory came from

        total_episodes = paddle_positions * episodes_per_position
        pbar = tqdm(total=total_episodes, desc="Generating trajectories")

        for position_idx in range(paddle_positions):
            # Calculate target x-position for this sweep position
            target_x = left_pos + (position_idx * step_size)

            for episode_idx in range(episodes_per_position):
                try:
                    # Reset environment
                    obs, info = self.env.reset()

                    # Move paddle all the way to the left
                    for _ in range(100):
                        self.env.step(self.LEFT)
                    actual_left = self.get_paddle_position()

                    # Calculate how many RIGHT steps needed from current position
                    if position_idx > 0:
                        # Move RIGHT until we reach target position
                        steps_taken = 0
                        max_attempts = 50
                        while steps_taken < max_attempts:
                            current_pos = self.get_paddle_position()
                            if current_pos is None:
                                break
                            if current_pos >= target_x:
                                break
                            self.env.step(self.RIGHT)
                            steps_taken += 1

                    # Get final paddle position before firing
                    final_paddle_pos = self.get_paddle_position()
                    if final_paddle_pos is None:
                        print(f"Warning: Could not detect paddle at position {position_idx}, episode {episode_idx}")
                        continue

                    # Verify we're at the expected position (within tolerance)
                    position_error = abs(final_paddle_pos - target_x)
                    if position_error > step_size:
                        print(f"Warning: Large position error at index {position_idx}: target={target_x}, actual={final_paddle_pos}, error={position_error}")

                    # Fire the ball
                    obs, reward, terminated, truncated = self.fire_ball()
                    if terminated or truncated:
                        print(f"Warning: Episode ended immediately after firing at position {position_idx}")
                        continue

                    # Collect trajectory with paddle policy
                    trajectory = self.collect_trajectory(
                        max_length=trajectory_length,
                        paddle_policy=policy
                    )
                    trajectory['paddle_start_position'] = final_paddle_pos
                    trajectory['position_index'] = position_idx
                    trajectory['target_position'] = target_x
                    trajectory['paddle_policy'] = paddle_policy

                    all_trajectories.append(trajectory)
                    paddle_position_map.append(position_idx)

                    pbar.set_postfix({
                        "position": position_idx,
                        "target_x": target_x,
                        "actual_x": final_paddle_pos,
                        "error": position_error,
                        "traj_len": trajectory['length'],
                        "reward": trajectory['rewards'].sum()
                    })
                    pbar.update(1)

                except Exception as e:
                    print(f"Error at position {position_idx}, episode {episode_idx}: {e}")
                    continue

        pbar.close()

        # Save dataset
        dataset = {
            'trajectories': all_trajectories,
            'paddle_position_map': paddle_position_map,
            'metadata': {
                'game': self.game_name,
                'paddle_positions': paddle_positions,
                'episodes_per_position': episodes_per_position,
                'trajectory_length': trajectory_length,
                'total_trajectories': len(all_trajectories),
                'paddle_policy': paddle_policy,  # Track which policy was used
            }
        }

        output_file = output_path / f"{self.game_name.lower()}_systematic_paddle_sweep.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"\n{'='*70}")
        print("DATASET GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total trajectories collected: {len(all_trajectories)}")
        print(f"Output file: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        # Statistics
        total_frames = sum(t['length'] for t in all_trajectories)
        total_reward = sum(t['rewards'].sum() for t in all_trajectories)
        avg_trajectory_length = total_frames / len(all_trajectories)
        avg_reward = total_reward / len(all_trajectories)

        print(f"\nStatistics:")
        print(f"  Total frames: {total_frames}")
        print(f"  Average trajectory length: {avg_trajectory_length:.1f}")
        print(f"  Total reward collected: {total_reward}")
        print(f"  Average reward per trajectory: {avg_reward:.2f}")

        # Paddle position coverage
        unique_positions = len(set(paddle_position_map))
        print(f"\nPaddle Position Coverage:")
        print(f"  Unique positions sampled: {unique_positions}/{paddle_positions}")

        position_counts = {}
        for pos in paddle_position_map:
            position_counts[pos] = position_counts.get(pos, 0) + 1

        print(f"  Position distribution:")
        for pos in sorted(position_counts.keys())[:5]:
            print(f"    Position {pos}: {position_counts[pos]} trajectories")
        if len(position_counts) > 5:
            print(f"    ... and {len(position_counts) - 5} more positions")

        print(f"{'='*70}\n")

        # Save heatmaps as visualizations
        self.save_heatmaps(output_path)

        self.env.close()
        return dataset

    def save_heatmaps(self, output_path):
        """Save paddle and ball heatmaps as PNG visualizations."""
        import matplotlib.pyplot as plt

        print(f"\n{'='*70}")
        print("GENERATING HEATMAP VISUALIZATIONS")
        print(f"{'='*70}")

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Paddle heatmap
        paddle_img = ax1.imshow(self.paddle_heatmap, cmap='hot', interpolation='nearest', aspect='auto')
        ax1.set_title(f'Paddle Position Heatmap\n(Total visits: {self.paddle_heatmap.sum():,})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position (pixels)', fontsize=12)
        ax1.set_ylabel('Y Position (pixels)', fontsize=12)
        ax1.grid(False)
        plt.colorbar(paddle_img, ax=ax1, label='Visit Count')

        # Ball heatmap
        ball_img = ax2.imshow(self.ball_heatmap, cmap='hot', interpolation='nearest', aspect='auto')
        ax2.set_title(f'Ball Position Heatmap\n(Total visits: {self.ball_heatmap.sum():,})', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X Position (pixels)', fontsize=12)
        ax2.set_ylabel('Y Position (pixels)', fontsize=12)
        ax2.grid(False)
        plt.colorbar(ball_img, ax=ax2, label='Visit Count')

        plt.tight_layout()

        # Save figure
        heatmap_file = output_path / f"{self.game_name.lower()}_heatmaps.png"
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Heatmaps saved to: {heatmap_file}")
        print(f"\nHeatmap Statistics:")
        print(f"  Paddle total visits: {self.paddle_heatmap.sum():,}")
        print(f"  Paddle max visits (hottest pixel): {self.paddle_heatmap.max():,}")
        print(f"  Paddle unique positions: {np.count_nonzero(self.paddle_heatmap):,}")
        print(f"  Ball total visits: {self.ball_heatmap.sum():,}")
        print(f"  Ball max visits (hottest pixel): {self.ball_heatmap.max():,}")
        print(f"  Ball unique positions: {np.count_nonzero(self.ball_heatmap):,}")

        # Save raw heatmap data as NPZ for later analysis
        heatmap_data_file = output_path / f"{self.game_name.lower()}_heatmaps.npz"
        np.savez_compressed(
            heatmap_data_file,
            paddle_heatmap=self.paddle_heatmap,
            ball_heatmap=self.ball_heatmap,
        )
        print(f"✅ Raw heatmap data saved to: {heatmap_data_file}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ATARI dataset with systematic paddle position sweeping"
    )
    parser.add_argument('--game', type=str, default='Breakout',
                       help='ATARI game name')
    parser.add_argument('--output_dir', type=str, default='data/systematic_sweep',
                       help='Output directory for dataset')
    parser.add_argument('--paddle_positions', type=int, default=None,
                       help='Number of paddle positions to sample (left to right). If not specified, auto-detects full range.')
    parser.add_argument('--episodes_per_position', type=int, default=3,
                       help='Number of episodes to collect at each paddle position')
    parser.add_argument('--trajectory_length', type=int, default=200,
                       help='Maximum length of each trajectory')
    parser.add_argument('--paddle_policy', type=str, default='static',
                       choices=['static', 'random', 'tracking', 'mixed'],
                       help='Paddle movement policy during trajectory collection. '
                            'static: paddle stays still (default). '
                            'random: random movements. '
                            'tracking: tracks ball (recommended for JEPA). '
                            'mixed: 70%% tracking + 30%% random (recommended for JEPA).')
    parser.add_argument('--mode', type=str, default='ram',
                       choices=['ram', 'vision'],
                       help='Observation mode. '
                            'ram: save RAM state (220 bytes, compact, default). '
                            'vision: save pixel frames (160x210 RGB, larger files).')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("SYSTEMATIC PADDLE SWEEP DATA GENERATION")
    print("="*70)
    print(f"Game: {args.game}")
    print(f"Output: {args.output_dir}")
    print(f"Configuration:")
    if args.paddle_positions is None:
        print(f"  - Paddle positions: AUTO-DETECT (full range)")
    else:
        print(f"  - Paddle positions to sample: {args.paddle_positions}")
    print(f"  - Episodes per position: {args.episodes_per_position}")
    if args.paddle_positions is not None:
        print(f"  - Total episodes: {args.paddle_positions * args.episodes_per_position}")
    print(f"  - Trajectory length: {args.trajectory_length}")
    print(f"  - Paddle policy: {args.paddle_policy}")
    print("="*70 + "\n")

    generator = SystematicPaddleDataGenerator(game_name=args.game, mode=args.mode)

    dataset = generator.generate_systematic_dataset(
        output_dir=args.output_dir,
        paddle_positions=args.paddle_positions,
        episodes_per_position=args.episodes_per_position,
        trajectory_length=args.trajectory_length,
        paddle_policy=args.paddle_policy,
    )

    print("✅ Dataset generation complete!")
    print(f"Use this dataset to train a world model with better trajectory coverage")


if __name__ == "__main__":
    main()
