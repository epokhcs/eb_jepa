"""
Test script to explore OCAtari object detection for Breakout.

This script:
1. Creates an OCAtari environment for Breakout
2. Runs a few episodes with random actions
3. Logs all detected objects and their properties
4. Tracks ball movement, player movement, and brick counts
"""

import sys
import random
from collections import defaultdict

try:
    from ocatari.core import OCAtari
except ImportError:
    print("ERROR: ocatari not installed. Install with: uv pip install ocatari")
    sys.exit(1)


def main():
    game_name = "Breakout"
    # Try different modes: "ram" for RAM extraction, "vision" for vision-based
    MODE = "ram"  # RAM extraction mode for Breakout
    HUD = False  # We don't need HUD objects for metrics

    print(f"Creating OCAtari environment for {game_name}...")
    print(f"Mode: {MODE}, HUD: {HUD}")
    env = OCAtari(game_name, mode=MODE, hud=HUD, render_mode='rgb_array')
    observation, info = env.reset()

    print(f"Action space: {env.action_space.n} actions")
    print(f"Observation shape: {observation.shape}")
    print()

    # Track statistics
    frame_count = 0
    episode_count = 0
    object_stats = defaultdict(int)

    # Track movements per episode
    ball_total_movement = 0
    player_total_movement = 0
    initial_brick_count = None
    current_brick_count = 0

    print("=" * 80)
    print("Starting test run (500 frames)...")
    print("=" * 80)

    for i in range(500):
        action = random.randint(0, env.action_space.n - 1)
        obs, reward, terminated, truncated, info = env.step(action)
        frame_count += 1

        # Every 10 frames, print detailed object info
        if i % 10 == 0:
            print(f"\n--- Frame {i} ---")
            print(f"Reward: {reward}")
            print(f"Detected objects: {len(env.objects)}")

            for obj in env.objects:
                # Skip NoObject entries
                if obj.category == "NoObject":
                    continue

                # Count object types
                object_stats[obj.category] += 1

                # Print object details
                print(f"  {obj.category}:")
                print(f"    Position: ({obj.x}, {obj.y})")
                print(f"    Size: {obj.w}x{obj.h}")
                print(f"    Movement: dx={obj.dx}, dy={obj.dy}")
                print(f"    Previous position: {obj.prev_xy}")
                print(f"    Center: {obj.center}")
                print(f"    HUD: {obj.hud}")

                # Track movements
                if obj.category.lower() == "ball":
                    movement = abs(obj.dx) + abs(obj.dy)
                    ball_total_movement += movement
                    print(f"    >>> Ball movement this frame: {movement} pixels")

                elif obj.category.lower() == "player":
                    movement = abs(obj.dx) + abs(obj.dy)
                    player_total_movement += movement
                    print(f"    >>> Player movement this frame: {movement} pixels")

                elif obj.category.lower() == "brick" or obj.category.lower() == "block":
                    current_brick_count += 1

            # Set initial brick count on first frame
            if initial_brick_count is None and current_brick_count > 0:
                initial_brick_count = current_brick_count
                print(f"\n  >>> Initial brick count: {initial_brick_count}")

            if current_brick_count > 0:
                bricks_broken = (initial_brick_count or 0) - current_brick_count
                print(f"  >>> Bricks broken so far: {bricks_broken}")

            current_brick_count = 0  # Reset for next frame

        if terminated or truncated:
            episode_count += 1
            print(f"\n{'=' * 80}")
            print(f"EPISODE {episode_count} ENDED (frame {i})")
            print(f"  Ball total movement: {ball_total_movement} pixels")
            print(f"  Player total movement: {player_total_movement} pixels")
            print(f"  Initial bricks: {initial_brick_count}")
            print(f"  Current bricks: {current_brick_count}")
            bricks_broken = (initial_brick_count or 0) - current_brick_count
            print(f"  Bricks broken: {bricks_broken}")
            print(f"{'=' * 80}\n")

            # Reset episode stats
            ball_total_movement = 0
            player_total_movement = 0
            initial_brick_count = None
            current_brick_count = 0

            observation, info = env.reset()

    env.close()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total frames: {frame_count}")
    print(f"Total episodes: {episode_count}")
    print("\nObject detection frequency:")
    for obj_type, count in sorted(object_stats.items()):
        print(f"  {obj_type}: {count} detections")

    print("\nObject categories found:")
    unique_categories = set(object_stats.keys())
    for cat in sorted(unique_categories):
        print(f"  - {cat}")


if __name__ == "__main__":
    main()
