from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Any, Dict
import torch


class TrajectoryBatch(NamedTuple):
    """
    Unified trajectory batch format across all environments.

    This format allows environment-specific data to be stored in metadata
    while keeping the core state/action interface consistent.
    """

    states: torch.Tensor  # [B, C, T, H, W] - observations
    actions: torch.Tensor  # [B, A, T] - actions (continuous or discrete)
    metadata: Dict[str, Any]  # Environment-specific metadata


@dataclass
class DatasetConfigBase:
    """Base configuration for all datasets."""

    size: int = 10000
    val_size: int = 10000
    batch_size: int = 128
    img_size: int = 64
    n_steps: int = 91
    sample_length: int = 17
    train: bool = True
    device: str = "cuda"
    normalize: bool = True


class DatasetBase(ABC, torch.utils.data.Dataset):
    """
    Abstract base class for all trajectory datasets.

    Subclasses must implement methods for generating or loading
    trajectory data in the unified TrajectoryBatch format.
    """

    @abstractmethod
    def __init__(self, config: DatasetConfigBase):
        """
        Initialize dataset with configuration.

        Args:
            config: Dataset configuration (subclass of DatasetConfigBase)
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> TrajectoryBatch:
        """
        Return a trajectory batch in unified format.

        Args:
            idx: Dataset index

        Returns:
            TrajectoryBatch with states, actions, and metadata
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of trajectory samples in the dataset."""
        pass

    @abstractmethod
    def get_env_specific_params(self) -> Dict[str, Any]:
        """
        Return environment-specific parameters needed for rendering.

        Examples:
            - Two Rooms: {'wall_x': ..., 'door_y': ...}
            - ATARI: {}

        Returns:
            Dictionary of environment-specific parameters
        """
        pass

    @property
    @abstractmethod
    def normalizer(self):
        """Return the normalizer for this dataset."""
        pass
