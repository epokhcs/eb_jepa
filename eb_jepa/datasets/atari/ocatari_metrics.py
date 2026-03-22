"""
OCAtari metrics tracking for Breakout and other Atari games.

This module provides functionality to:
1. Track ball movement (total pixels moved)
2. Track player movement (total pixels moved)
3. Track bricks broken (by counting Block objects)

Usage:
    from eb_jepa.datasets.atari.ocatari_metrics import OCAtariMetricsTracker

    tracker = OCAtariMetricsTracker(game_name="Breakout")
    tracker.reset()

    for step in range(num_steps):
        action = agent.get_action()
        tracker.step(action)

    metrics = tracker.get_metrics()
    print(f"Ball movement: {metrics['ball_total_movement']} pixels")
    print(f"Player movement: {metrics['player_total_movement']} pixels")
    print(f"Bricks broken: {metrics['bricks_broken']}")
"""

import numpy as np
from typing import Dict, Optional, Any

try:
    from ocatari.core import OCAtari
    OCATARI_AVAILABLE = True
except ImportError:
    OCATARI_AVAILABLE = False


class OCAtariMetricsTracker:
    """
    Tracks metrics from OCAtari object detection for Atari games.

    For Breakout, tracks:
    - Ball total pixel movement (dx + dy over all frames)
    - Player (paddle) total pixel movement
    - Bricks broken (by counting Block objects)
    """

    def __init__(
        self,
        game_name: str = "Breakout",
        mode: str = "ram",
        hud: bool = False,
        render_mode: str = "rgb_array",
    ):
        if not OCATARI_AVAILABLE:
            raise ImportError(
                "OCAtari is not installed. Install with: pip install ocatari"
            )

        self.game_name = game_name
        self.env = OCAtari(game_name, mode=mode, hud=hud, render_mode=render_mode)

        # Metrics
        self.ball_total_movement = 0
        self.player_total_movement = 0
        self.initial_brick_count = None
        self.current_brick_count = 0

        # State
        self.step_count = 0
        self.episode_count = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and metrics."""
        obs, info = self.env.reset()

        # Reset episode-level metrics
        self.ball_total_movement = 0
        self.player_total_movement = 0
        self.initial_brick_count = None
        self.current_brick_count = 0
        self.step_count = 0

        return obs

    def step(self, action: int) -> tuple:
        """
        Take a step in the environment and update metrics.

        Returns:
            obs, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Reset current brick count for this frame
        self.current_brick_count = 0

        # Update metrics from detected objects
        for obj in self.env.objects:
            # Skip NoObject entries
            if obj.category == "NoObject":
                continue

            # Track ball movement
            if obj.category.lower() == "ball":
                movement = abs(obj.dx) + abs(obj.dy)
                self.ball_total_movement += movement

            # Track player movement
            elif obj.category.lower() == "player":
                movement = abs(obj.dx) + abs(obj.dy)
                self.player_total_movement += movement

            # Track bricks (Block objects in Breakout)
            elif obj.category.lower() in ("brick", "block"):
                self.current_brick_count += 1

        # Set initial brick count on first detection
        if self.initial_brick_count is None and self.current_brick_count > 0:
            self.initial_brick_count = self.current_brick_count

        # Handle episode end
        if terminated or truncated:
            self.episode_count += 1

        return obs, reward, terminated, truncated, info

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dictionary with:
            - ball_total_movement: Total pixels moved by ball
            - player_total_movement: Total pixels moved by player
            - bricks_broken: Number of bricks destroyed
            - initial_bricks: Initial number of brick blocks
            - current_bricks: Current number of brick blocks
            - step_count: Number of steps in this episode
            - episode_count: Total number of episodes
        """
        bricks_broken = 0
        if self.initial_brick_count is not None:
            bricks_broken = self.initial_brick_count - self.current_brick_count

        return {
            "ball_total_movement": self.ball_total_movement,
            "player_total_movement": self.player_total_movement,
            "bricks_broken": bricks_broken,
            "initial_bricks": self.initial_brick_count or 0,
            "current_bricks": self.current_brick_count,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
        }

    def close(self):
        """Close the OCAtari environment."""
        self.env.close()


class BatchOCAtariMetricsTracker:
    """
    Tracks OCAtari metrics for a batch of parallel environments.

    Note: This creates independent OCAtari environments for tracking.
    It does NOT wrap the existing gymnasium environments.
    """

    def __init__(
        self,
        game_name: str = "Breakout",
        num_envs: int = 1,
        mode: str = "ram",
        hud: bool = False,
    ):
        if not OCATARI_AVAILABLE:
            raise ImportError(
                "OCAtari is not installed. Install with: pip install ocatari"
            )

        self.trackers = [
            OCAtariMetricsTracker(game_name, mode, hud)
            for _ in range(num_envs)
        ]

    def reset(self, env_idx: Optional[int] = None):
        """Reset one or all environments."""
        if env_idx is not None:
            return self.trackers[env_idx].reset()
        else:
            return [tracker.reset() for tracker in self.trackers]

    def step(self, actions: np.ndarray):
        """
        Take a step in all environments.

        Args:
            actions: Array of shape (num_envs,) with actions for each environment
        """
        results = []
        for i, action in enumerate(actions):
            result = self.trackers[i].step(int(action))
            results.append(result)
        return results

    def get_metrics(self, aggregate: bool = True) -> Dict[str, Any]:
        """
        Get metrics from all environments.

        Args:
            aggregate: If True, return aggregated metrics (sum/mean).
                      If False, return list of per-env metrics.

        Returns:
            If aggregate=True:
                Dictionary with summed/mean metrics across all envs
            If aggregate=False:
                List of dictionaries, one per environment
        """
        all_metrics = [tracker.get_metrics() for tracker in self.trackers]

        if not aggregate:
            return all_metrics

        # Aggregate metrics
        aggregated = {
            "ball_total_movement": sum(m["ball_total_movement"] for m in all_metrics),
            "player_total_movement": sum(m["player_total_movement"] for m in all_metrics),
            "bricks_broken": sum(m["bricks_broken"] for m in all_metrics),
            "initial_bricks": sum(m["initial_bricks"] for m in all_metrics),
            "current_bricks": sum(m["current_bricks"] for m in all_metrics),
            "step_count": sum(m["step_count"] for m in all_metrics),
            "episode_count": sum(m["episode_count"] for m in all_metrics),
        }

        return aggregated

    def close(self):
        """Close all OCAtari environments."""
        for tracker in self.trackers:
            tracker.close()
