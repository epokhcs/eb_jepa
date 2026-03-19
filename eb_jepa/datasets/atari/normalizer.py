import torch

from ..base import NormalizerBase


class AtariNormalizer(NormalizerBase):
    """
    Normalizer for ATARI environments.

    ATARI observations are pixel values in [0, 255] that need to be
    normalized to [0, 1] for neural network processing.
    """

    def __init__(self):
        super().__init__()
        self.pixel_mean = 0.0
        self.pixel_std = 255.0

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel observations from [0, 255] to [0, 1].

        Args:
            state: Raw observation tensor with values in [0, 255]

        Returns:
            Normalized observation tensor with values in [0, 1]
        """
        return state.float() / 255.0

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize pixel observations from [0, 1] to [0, 255].

        Args:
            state: Normalized observation tensor with values in [0, 1]

        Returns:
            Raw observation tensor with values in [0, 255]
        """
        return (state * 255.0).clamp(0, 255)

    def normalize_latent_state(self, latent_state):
        """
        For ATARI, latent state normalization is identity.

        Unlike Two Rooms which has explicit 2D positions, ATARI's latent
        state is the learned embedding, which doesn't need normalization.
        """
        return latent_state

    def unnormalize_latent_state(self, latent_state):
        """
        For ATARI, latent state unnormalization is identity.
        """
        return latent_state

    def unnormalize_mse(self, mse: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize MSE for interpretable metrics.

        For pixel-space MSE:
        MSE_normalized = (1/255^2) * MSE_raw
        MSE_raw = 255^2 * MSE_normalized
        """
        return mse * (self.pixel_std ** 2)
