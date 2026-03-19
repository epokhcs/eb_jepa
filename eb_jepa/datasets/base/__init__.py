"""
Base abstractions for environment and dataset interfaces.

This module provides abstract base classes that define the contract
for environments and datasets in the ac_video_jepa framework.
"""

from .env_base import EnvBase
from .dataset_base import DatasetBase, DatasetConfigBase, TrajectoryBatch
from .normalizer_base import NormalizerBase

__all__ = [
    "EnvBase",
    "DatasetBase",
    "DatasetConfigBase",
    "TrajectoryBatch",
    "NormalizerBase",
]
