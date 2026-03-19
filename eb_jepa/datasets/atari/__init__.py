"""
ATARI environment and dataset registration.
"""

from ..registry import EnvironmentRegistry
from .env import AtariEnv
from .dataset import AtariDataset
from .config import AtariConfig

# Register ATARI environment in the global registry
EnvironmentRegistry.register_env(
    name="atari",
    env_class=AtariEnv,
    dataset_class=AtariDataset,
    config_class=AtariConfig,
)

__all__ = ["AtariEnv", "AtariDataset", "AtariConfig"]
