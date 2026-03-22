"""
Paddle movement strategies for systematic sweep data collection.

This document explains different options for paddle control during trajectory collection.
"""

import numpy as np


class PaddlePolicy:
    """Base class for paddle movement policies."""

    def __init__(self, env):
        self.env = env
        self.NOOP = 0
        self.FIRE = 1
        self.RIGHT = 2
        self.LEFT = 3

    def get_action(self, obs, info):
        """Return action based on current state."""
        raise NotImplementedError


class StaticPaddlePolicy(PaddlePolicy):
    """
    Paddle stays static after firing (current approach).

    Pros:
    - Simple dynamics to learn
    - Focused on ball physics

    Cons:
    - Not learning action-conditioned dynamics
    - Paddle position irrelevant after firing
    - Can't plan paddle movements
    """

    def get_action(self, obs, info):
        return self.NOOP


class RandomPaddlePolicy(PaddlePolicy):
    """
    Paddle moves randomly (LEFT/RIGHT/NOOP).

    Pros:
    - Diverse action sequences
    - Learns paddle-ball interaction
    - Good for action-conditioned JEPA

    Cons:
    - Inefficient (misses ball often)
    - May lead to early episode termination
    - Not realistic gameplay
    """

    def __init__(self, env, p_move=0.3):
        super().__init__(env)
        self.p_move = p_move  # Probability of moving vs NOOP

    def get_action(self, obs, info):
        if np.random.random() < self.p_move:
            return np.random.choice([self.LEFT, self.RIGHT])
        else:
            return self.NOOP


class BallTrackingPolicy(PaddlePolicy):
    """
    Paddle tries to track ball's x-position using OCAtari.

    Pros:
    - Realistic gameplay
    - Learns catching behavior
    - Longer episodes (ball doesn't fall off)
    - Best for eventual RL/planning

    Cons:
    - More complex to implement
    - Requires OCAtari object detection
    - May be "too good" (less diversity?)
    """

    def get_action(self, obs, info):
        # Get paddle and ball positions from OCAtari
        paddle_x = None
        ball_x = None

        for obj in self.env.objects:
            if obj.category.lower() == "player":
                paddle_x = obj.x
            elif obj.category.lower() == "ball":
                ball_x = obj.x

        if paddle_x is None or ball_x is None:
            return self.NOOP

        # Move paddle toward ball
        if ball_x < paddle_x - 5:  # Ball is left of paddle
            return self.LEFT
        elif ball_x > paddle_x + 5:  # Ball is right of paddle
            return self.RIGHT
        else:
            return self.NOOP


class MixedPolicy(PaddlePolicy):
    """
    Combination: track ball with some random noise.

    Pros:
    - Balances realism and diversity
    - Learns from both good and bad paddle movements
    - Good action variation

    Cons:
    - More hyperparameters to tune
    """

    def __init__(self, env, noise_prob=0.2):
        super().__init__(env)
        self.tracking_policy = BallTrackingPolicy(env)
        self.random_policy = RandomPaddlePolicy(env)
        self.noise_prob = noise_prob

    def get_action(self, obs, info):
        if np.random.random() < self.noise_prob:
            return self.random_policy.get_action(obs, info)
        else:
            return self.tracking_policy.get_action(obs, info)


class PeriodicPolicy(PaddlePolicy):
    """
    Paddle oscillates left-right periodically.

    Pros:
    - Predictable but active
    - Guaranteed action diversity
    - Covers full horizontal range

    Cons:
    - Unrealistic behavior
    - May not correlate well with ball position
    """

    def __init__(self, env, period=10):
        super().__init__(env)
        self.step_count = 0
        self.period = period

    def get_action(self, obs, info):
        self.step_count += 1
        # Alternate between LEFT and RIGHT every 'period' steps
        if (self.step_count // self.period) % 2 == 0:
            return self.RIGHT
        else:
            return self.LEFT


# Recommendation matrix for different use cases
RECOMMENDATIONS = {
    "passive_world_model": {
        "policy": "StaticPaddlePolicy",
        "reason": "If you only want to learn ball physics, not action dynamics"
    },
    "action_conditioned_jepa": {
        "policy": "BallTrackingPolicy or MixedPolicy",
        "reason": "JEPA needs to learn how actions affect future states. Tracking + noise gives best diversity while maintaining realism."
    },
    "eventual_rl_planning": {
        "policy": "BallTrackingPolicy",
        "reason": "Planning needs to learn catching behavior. Model must understand paddle-ball interaction."
    },
    "maximum_diversity": {
        "policy": "RandomPaddlePolicy",
        "reason": "Explores widest range of action sequences, but inefficient."
    },
    "balanced_approach": {
        "policy": "MixedPolicy(noise_prob=0.3)",
        "reason": "70% tracking + 30% random gives good coverage with reasonable episode length."
    }
}


def print_recommendations():
    """Print policy recommendations."""
    print("=" * 70)
    print("PADDLE POLICY RECOMMENDATIONS")
    print("=" * 70)
    for use_case, rec in RECOMMENDATIONS.items():
        print(f"\n{use_case.upper().replace('_', ' ')}:")
        print(f"  Policy: {rec['policy']}")
        print(f"  Reason: {rec['reason']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_recommendations()
