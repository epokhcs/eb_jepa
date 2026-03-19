from abc import ABC, abstractmethod
from typing import Any
import torch


class NormalizerBase(ABC):
    """
    Abstract base class for state/action normalization.

    Different environments may require different normalization strategies:
    - Two Rooms: Normalize 2D positions and pixel values
    - ATARI: Normalize pixel values (0-255 -> 0-1)
    """

    @abstractmethod
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize observation/state tensor.

        Args:
            state: Raw observation tensor

        Returns:
            Normalized observation tensor
        """
        pass

    @abstractmethod
    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize observation/state tensor.

        Args:
            state: Normalized observation tensor

        Returns:
            Raw observation tensor
        """
        pass

    @abstractmethod
    def normalize_latent_state(self, latent_state: Any) -> Any:
        """
        Normalize latent state representation (e.g., positions, features).

        Examples:
            - Two Rooms: normalize (x, y) coordinates
            - ATARI: may be identity or normalize game score

        Args:
            latent_state: Raw latent state (format depends on environment)

        Returns:
            Normalized latent state
        """
        pass

    @abstractmethod
    def unnormalize_latent_state(self, latent_state: Any) -> Any:
        """
        Unnormalize latent state representation.

        Args:
            latent_state: Normalized latent state

        Returns:
            Raw latent state
        """
        pass

    @abstractmethod
    def unnormalize_mse(self, mse: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize MSE for interpretable metrics.

        This is useful for computing distances in the original space
        rather than the normalized space.

        Args:
            mse: MSE in normalized space

        Returns:
            MSE in original space
        """
        pass
