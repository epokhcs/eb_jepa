from dataclasses import dataclass
from typing import Optional

from ..base import DatasetConfigBase


@dataclass
class AtariConfig(DatasetConfigBase):
    """Configuration for ATARI environment and dataset."""

    # Game settings
    game_name: str = "Breakout"
    frame_stack: int = 1  # Number of frames to stack
    frame_skip: int = 4  # Frame skip (action repeat)
    grayscale: bool = True  # Use grayscale observations

    # Observation settings
    img_size: int = 84  # Standard ATARI resolution (will resize to this)
    dobs: int = 1  # Number of observation channels (1 for grayscale, 3 for RGB)

    # Action settings
    action_repeat: int = 1  # Additional action repeat on top of frame_skip

    # Dataset settings
    size: int = 10000
    val_size: int = 1000
    batch_size: int = 32
    n_steps: int = 200  # Length of trajectories to generate
    sample_length: int = 17  # Length of samples for training

    # Environment settings
    normalize: bool = True
    device: str = "cuda"
    train: bool = True

    # Evaluation settings
    score_based_success: bool = True
    success_score_threshold: float = 20.0  # Score difference threshold for success

    # Data collection
    random_policy: bool = True  # Use random policy for data collection
    num_workers: int = 0
    pin_mem: bool = False
    persistent_workers: bool = False
