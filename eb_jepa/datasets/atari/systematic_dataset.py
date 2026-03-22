import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, Any

from ..base import DatasetBase, TrajectoryBatch
from .config import AtariConfig
from .normalizer import AtariNormalizer


class SystematicAtariDataset(DatasetBase):
    """
    ATARI dataset that loads pre-collected systematic paddle sweep trajectories.

    This dataset uses trajectories collected with systematic paddle position variation,
    ensuring uniform coverage of initial conditions and better world model training.
    """

    def __init__(self, config: AtariConfig, dataset_path: str):
        """
        Initialize systematic ATARI dataset.

        Args:
            config: AtariConfig with dataset settings
            dataset_path: Path to pickle file containing systematic sweep data
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

        # Load dataset
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        print(f"Loading systematic sweep dataset from: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)

        self.trajectories = self.dataset['trajectories']
        self.metadata = self.dataset['metadata']

        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Game: {self.metadata['game']}")
        print(f"Paddle positions: {self.metadata['paddle_positions']}")
        print(f"Episodes per position: {self.metadata['episodes_per_position']}")

        # Initialize normalizer
        if config.normalize:
            self._normalizer = AtariNormalizer()
        else:
            self._normalizer = None

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> TrajectoryBatch:
        """
        Return a trajectory sample from the pre-collected dataset.

        Args:
            idx: Dataset index

        Returns:
            TrajectoryBatch with states, actions, and metadata
        """
        # Get trajectory
        trajectory = self.trajectories[idx % len(self.trajectories)]

        # Extract data
        observations = trajectory['observations']  # [T, H, W, C]
        actions = trajectory['actions']  # [T]
        rewards = trajectory['rewards']  # [T]

        # Sample a subsequence if needed
        T = len(observations)
        if T > self.config.sample_length:
            start_idx = np.random.randint(0, T - self.config.sample_length + 1)
            end_idx = start_idx + self.config.sample_length

            observations = observations[start_idx:end_idx]
            actions = actions[start_idx:end_idx]
            rewards = rewards[start_idx:end_idx]

        # Convert observations to grayscale and resize to 84x84
        states = []
        for obs in observations:
            # Convert RGB to grayscale if needed
            if obs.shape[-1] == 3:  # RGB
                gray = 0.299 * obs[:, :, 0] + 0.587 * obs[:, :, 1] + 0.114 * obs[:, :, 2]
            else:
                gray = obs[:, :, 0] if len(obs.shape) == 3 else obs

            # Ensure numpy array with correct dtype
            gray = np.asarray(gray, dtype=np.float32)

            # Resize to 84x84 if needed
            if gray.shape != (84, 84):
                import cv2
                gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

            # Normalize to [0, 1]
            if gray.max() > 1.0:
                gray = gray / 255.0

            states.append(torch.from_numpy(gray).float())

        # Convert to proper format
        # states: [T, H, W] -> [1, T, H, W] (grayscale channel)
        states = torch.stack(states).unsqueeze(0)

        # actions: [T] -> [1, T] (discrete actions)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)

        # Create metadata
        metadata = {
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "episode_idx": idx,
            "paddle_start_position": trajectory.get('paddle_start_position', None),
            "position_index": trajectory.get('position_index', None),
        }

        return TrajectoryBatch(states=states, actions=actions, metadata=metadata)

    def get_env_specific_params(self) -> Dict[str, Any]:
        """
        Return environment-specific parameters.
        """
        return {
            "game_name": self.metadata['game'],
            "action_space_size": 4,  # Breakout has 4 actions
            "total_trajectories": len(self.trajectories),
            "paddle_positions": self.metadata['paddle_positions'],
        }

    @property
    def normalizer(self):
        """Return the normalizer for this dataset."""
        return self._normalizer
