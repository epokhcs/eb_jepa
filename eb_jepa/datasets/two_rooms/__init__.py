"""
Two Rooms environment and dataset registration.
"""

from ..registry import EnvironmentRegistry
from .env import DotWall
from .wall_dataset import WallDataset, WallDatasetConfig

# Register Two Rooms environment in the global registry
EnvironmentRegistry.register_env(
    name="two_rooms",
    env_class=DotWall,
    dataset_class=WallDataset,
    config_class=WallDatasetConfig,
)

__all__ = ["DotWall", "WallDataset", "WallDatasetConfig"]
