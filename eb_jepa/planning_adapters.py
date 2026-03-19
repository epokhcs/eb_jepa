"""
Planning adapters to bridge environment-specific functionality with planning code.

These adapters handle environment-specific operations like rendering latent
representations to pixel space and extracting goal states from info dicts.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Any, Dict, Optional


class PlanningAdapter(ABC):
    """
    Adapter to bridge environment-specific rendering and evaluation.

    This separates environment physics from planning visualization,
    allowing the same planning code to work across different environments.
    """

    def __init__(self, env, prober=None, normalizer=None):
        """
        Initialize planning adapter.

        Args:
            env: Environment instance (implementing EnvBase)
            prober: Probe head for decoding latents to interpretable states
            normalizer: Normalizer for state/latent normalization
        """
        self.env = env
        self.prober = prober
        self.normalizer = normalizer

    @abstractmethod
    def decode_latent_to_pixel(
        self, predicted_encs: torch.Tensor, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Decode latent encodings to pixel frames for visualization.

        Args:
            predicted_encs: [B, D, T, H, W] latent encodings from model
            metadata: Environment-specific metadata

        Returns:
            frames: [B, T, H, W, C] uint8 numpy array
        """
        pass

    @abstractmethod
    def extract_goal_state(self, info: Dict) -> Any:
        """
        Extract goal state from environment info dictionary.

        Args:
            info: Info dict returned by environment

        Returns:
            Goal state in environment-specific format
        """
        pass


class TwoRoomsPlanningAdapter(PlanningAdapter):
    """Planning adapter for Two Rooms environment."""

    def decode_latent_to_pixel(
        self, predicted_encs: torch.Tensor, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Decode latents to pixel frames using position probe head.

        For Two Rooms, we:
        1. Use probe head to predict 2D positions from latents
        2. Unnormalize positions
        3. Render positions as pixel frames using env's coord_to_pixel

        Args:
            predicted_encs: [B, D, T, H, W] latent encodings
            metadata: Dict with 'wall_x' and 'door_y' keys

        Returns:
            frames: [B, T, H, W, C] uint8 numpy array
        """
        if self.prober is None:
            raise ValueError("TwoRoomsPlanningAdapter requires a prober")

        B, D, T, H, W = predicted_encs.shape

        # Decode positions from latents: [B, D, T, H, W] -> [B, 2, T]
        positions = self.prober.apply_head(predicted_encs).permute(0, 2, 1).cpu()

        # Unnormalize positions to original coordinate space
        if self.normalizer is not None:
            positions = self.normalizer.unnormalize_latent_state(positions)

        # Render frames using environment's coord_to_pixel method
        wall_x = metadata.get("wall_x")
        door_y = metadata.get("door_y")

        frames = self.env.coord_to_pixel(positions, wall_x=wall_x, door_y=door_y)

        # Convert to [B, T, H, W, C] format
        return frames.permute(0, 1, 3, 4, 2).cpu().numpy()

    def extract_goal_state(self, info: Dict) -> Any:
        """
        Extract goal position from info dict.

        Args:
            info: Info dict with 'target_position' key

        Returns:
            Goal position as [2] tensor (x, y coordinates)
        """
        return info.get("target_position")


class AtariPlanningAdapter(PlanningAdapter):
    """Planning adapter for ATARI environment."""

    def __init__(self, env, prober=None, normalizer=None, decoder=None):
        """
        Initialize ATARI planning adapter.

        Args:
            env: ATARI environment instance
            prober: Optional probe head (may not be needed for ATARI)
            normalizer: Normalizer for state normalization
            decoder: Decoder network to map latents to pixel space
        """
        super().__init__(env, prober, normalizer)
        self.decoder = decoder

    def decode_latent_to_pixel(
        self, predicted_encs: torch.Tensor, metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Decode latents to pixel frames using decoder network.

        For ATARI, we need a learned decoder to map latent representations
        back to pixel space since there's no explicit state like positions.

        Args:
            predicted_encs: [B, D, T, H, W] latent encodings
            metadata: Dict (may contain rewards, dones, etc.)

        Returns:
            frames: [B, T, H, W, C] uint8 numpy array
        """
        if self.decoder is None:
            raise NotImplementedError(
                "ATARI planning adapter requires a decoder network. "
                "The decoder should be trained jointly with the world model "
                "to reconstruct pixel observations from latent representations."
            )

        # Decode latents to pixels
        # decoder: [B, D, T, H, W] -> [B, C, T, H, W]
        decoded_frames = self.decoder(predicted_encs)

        # Unnormalize pixels (0-1 -> 0-255)
        if self.normalizer is not None:
            decoded_frames = self.normalizer.unnormalize_state(decoded_frames)

        # Convert to [B, T, H, W, C] uint8 numpy array
        decoded_frames = decoded_frames.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        decoded_frames = decoded_frames.clamp(0, 255).to(torch.uint8)

        return decoded_frames.cpu().numpy()

    def extract_goal_state(self, info: Dict) -> Any:
        """
        Extract goal state from info dict.

        For ATARI, the goal might be:
        - A target score
        - A target frame/observation
        - A specific game state

        Args:
            info: Info dict from environment

        Returns:
            Goal state (score, observation, or other representation)
        """
        # Try to get target score first
        if "target_score" in info:
            return info["target_score"]

        # Fall back to target observation
        if "target_obs" in info:
            return info["target_obs"]

        # No explicit goal for ATARI in standard setup
        return None
