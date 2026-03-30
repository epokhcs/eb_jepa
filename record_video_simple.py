#!/usr/bin/env python3
"""
Simple script to record MPPI planning video.
Uses the same checkpoint loading as test_planning_with_rewards.py which works.
"""
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import imageio

sys.path.insert(0, str(Path(__file__).parent))

from eb_jepa.checkpoint_utils import load_jepa_from_checkpoint
from eb_jepa.datasets.atari.env import AtariEnv
from eb_jepa.datasets.atari.config import AtariConfig
from eb_jepa.planning import MPPIPlanner

def record_video(checkpoint_path, output_path="videos/breakout_mppi.mp4", num_steps=200):
    """Record video of MPPI planner playing Breakout."""

    # Load checkpoint
    print("Loading checkpoint...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Get config path from checkpoint directory
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.yaml"

    jepa, reward_head, config = load_jepa_from_checkpoint(
        checkpoint_path,
        str(config_path),
        device=str(device)
    )
    jepa.eval()
    print(f"✅ Model loaded on {device}")

    # Create environment
    print("Creating environment...")
    env_config = AtariConfig(game_name="Breakout", device="cpu", grayscale=True, img_size=84)
    env = AtariEnv(env_config)

    # Create planner
    print("Creating MPPI planner...")
    planner = MPPIPlanner(
        num_actions=4,
        plan_length=16,
        num_samples=100,
        n_iters=10,
        temperature=1.0,
        var_scale=1.0
    )

    # Initialize planning objective for reward prediction
    from eb_jepa.planning import ReprTargetDistMPCObjective
    objective = ReprTargetDistMPCObjective(
        objective_type="reward",
        sum_all_diffs=True,
        alpha=0.0
    )

    # Run and record
    print(f"\n🎬 Recording {num_steps} steps...")
    frames = []
    obs, info = env.reset()

    # Get RGB frame for video
    rgb_frame = env.env.render()
    frames.append(rgb_frame)

    total_reward = 0
    pbar = tqdm(range(num_steps), desc="Recording")

    for step in pbar:
        # Prepare observation (grayscale)
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)  # [1, C, H, W]

        # Plan action using MPPI
        with torch.no_grad():
            result = planner.plan(
                model=jepa,
                obs=obs_tensor,
                objective_fn=objective,
                reward_head=reward_head,
                ctxt_window_time=1,
                device=device
            )

        action = int(result.action.cpu().item())

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Get RGB frame
        rgb_frame = env.env.render()
        frames.append(rgb_frame)

        # Update progress bar
        pbar.set_postfix({"reward": total_reward, "action": action})

        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            break

    pbar.close()
    env.close()

    # Save video
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving video to {output_path}...")
    imageio.mimsave(str(output_path), frames, fps=30)

    print(f"\n✅ Video saved!")
    print(f"📊 Total reward: {total_reward}")
    print(f"📹 Episode length: {len(frames)} frames ({len(frames)/30:.1f} seconds at 30fps)")
    print(f"📁 Video path: {output_path.absolute()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="videos/breakout_mppi.mp4", help="Output video path")
    parser.add_argument("--steps", type=int, default=200, help="Max steps to record")
    args = parser.parse_args()

    record_video(args.checkpoint, args.output, args.steps)
