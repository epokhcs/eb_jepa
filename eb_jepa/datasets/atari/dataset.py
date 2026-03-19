import numpy as np
import torch
from typing import Dict, Any

from ..base import DatasetBase, TrajectoryBatch
from .config import AtariConfig
from .env import AtariEnv
from .normalizer import AtariNormalizer


class AtariDataset(DatasetBase):
    """
    ATARI dataset that generates trajectories on-the-fly.

    Collects trajectories using a random policy (or provided policy)
    and returns them in TrajectoryBatch format.
    """

    def __init__(self, config: AtariConfig):
        """
        Initialize ATARI dataset.

        Args:
            config: AtariConfig with dataset settings
        """
        self.config = config

        # Setup device with auto-detection (CUDA > MPS > CPU)
        if config.device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # Update config device to match detected device
        self.config.device = str(self.device)

        # Create environment for data collection
        self.env = AtariEnv(config)

        # Initialize normalizer
        if config.normalize:
            self._normalizer = AtariNormalizer()
        else:
            self._normalizer = None

        # Cache for generated trajectories (optional for efficiency)
        self._trajectory_cache = {}

    def __len__(self) -> int:
        """Return dataset size."""
        return self.config.size

    def __getitem__(self, idx: int) -> TrajectoryBatch:
        """
        Generate and return a trajectory sample.

        Args:
            idx: Dataset index

        Returns:
            TrajectoryBatch with states, actions, and metadata
        """
        # Generate a new trajectory
        trajectory = self._generate_trajectory()

        # Extract states and actions
        states = trajectory["states"]  # [T, C, H, W]
        actions = trajectory["actions"]  # [T]
        rewards = trajectory["rewards"]  # [T]
        dones = trajectory["dones"]  # [T]

        # Sample a subsequence of length sample_length
        if len(states) > self.config.sample_length:
            start_idx = np.random.randint(0, len(states) - self.config.sample_length + 1)
            end_idx = start_idx + self.config.sample_length

            states = states[start_idx:end_idx]
            actions = actions[start_idx:end_idx]
            rewards = rewards[start_idx:end_idx]
            dones = dones[start_idx:end_idx]

        # Convert to proper format
        # states: [T, C, H, W] -> [C, T, H, W]
        states = torch.stack(states).permute(1, 0, 2, 3)

        # actions: [T] -> [1, T] (discrete actions)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)

        # Create metadata
        metadata = {
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.bool),
            "episode_idx": idx,
        }

        return TrajectoryBatch(states=states, actions=actions, metadata=metadata)

    def _generate_trajectory(self) -> Dict[str, Any]:
        """
        Generate a trajectory using random policy.

        Returns:
            Dictionary with states, actions, rewards, dones
        """
        states = []
        actions = []
        rewards = []
        dones = []

        # Reset environment
        obs, info = self.env.reset()
        states.append(obs)

        # Collect trajectory
        for step in range(self.config.n_steps):
            # Random policy: sample random action
            action = self.env.env.action_space.sample()
            actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            states.append(obs)
            rewards.append(reward)
            done = terminated or truncated
            dones.append(done)

            # Reset if episode ends
            if done and step < self.config.n_steps - 1:
                obs, info = self.env.reset()
                states[-1] = obs  # Replace last state with reset state

        # Remove last state (we have n_steps actions and n_steps+1 states)
        states = states[:-1]

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def get_env_specific_params(self) -> Dict[str, Any]:
        """
        Return environment-specific parameters.

        For ATARI, there are no special rendering parameters like
        Two Rooms' wall positions.
        """
        return {
            "game_name": self.config.game_name,
            "action_space_size": self.env.get_action_space_info()["n"],
        }

    @property
    def normalizer(self):
        """Return the normalizer for this dataset."""
        return self._normalizer

    def __del__(self):
        """Cleanup environment on deletion."""
        if hasattr(self, 'env'):
            self.env.close()
