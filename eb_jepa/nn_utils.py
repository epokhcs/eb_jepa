"""Shared utilities for neural network initialization and common patterns."""

import torch.nn as nn
from einops import rearrange
import torch
import numpy as np


def init_module_weights(m, std: float = 0.02):
    """
    Initialize weights for common layer types using truncated normal distribution.

    This is a unified weight initialization function used across the codebase.
    Apply it via module.apply(init_module_weights) or as a method wrapper.

    Args:
        m: PyTorch module to initialize
        std: Standard deviation for truncated normal initialization (default: 0.02)
    """
    if isinstance(
        m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    ):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TemporalBatchMixin:
    """
    Mixin class that handles automatic temporal batching for 4D/5D tensors.

    This mixin provides a unified forward() method that:
    - For 5D tensors [B, C, T, H, W]: flattens temporal dim, applies _forward(), restores shape
    - For 4D tensors [B, C, H, W]: directly applies _forward()

    Subclasses must implement _forward(self, x) for 4D tensors.
    """

    def _forward(self, x):
        """
        Process 4D tensor [B, C, H, W]. Must be implemented by subclasses.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C_out, H_out, W_out]
        """
        raise NotImplementedError("Subclasses must implement _forward()")

    def forward(self, x):
        """
        Forward pass supporting both 4D and 5D tensors.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, C, T, H, W]

        Returns:
            Output tensor with same batch and temporal dimensions as input
        """
        assert x.ndim in [
            4,
            5,
        ], "Supports only 4D [B, C, H, W] or 5D [B, C, T, H, W] tensors"
        if x.ndim == 5:
            b = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            out = rearrange(out, "(b t) c h w -> b c t h w", b=b)
            return out
        else:
            return self._forward(x)


def to_model_obs(obs, device=None):
    """
    Convert a numpy array or tensor observation to model input shape [B, C, T, H, W] and device.
    Accepts [H, W], [C, H, W], [B, C, H, W], or [B, C, T, H, W].
    Adds batch/time dims as needed.
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    if obs.ndim == 2:
        obs = obs.unsqueeze(0)  # [1, H, W]
    if obs.ndim == 3:
        if obs.shape[-1] in [1, 3]:
            obs = obs.permute(2, 0, 1)  # [C, H, W]
        obs = obs.unsqueeze(0)  # [1, C, H, W]
    if obs.ndim == 4:
        obs = obs.unsqueeze(2)  # [B, C, 1, H, W]
    assert obs.ndim == 5, f"Expected obs to have 5 dims, got {obs.shape}"
    if device is not None:
        obs = obs.to(device)
    return obs


def to_model_actions(actions, device=None):
    """
    Convert a numpy array or tensor of actions to model input shape [B, 1, T] and device.
    Accepts [T], [1, T], or [B, 1, T].
    """
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    # Debug: print and assert for negative actions
    print("[to_model_actions] actions (pre-shape):", actions)
    if (actions < 0).any():
        print("[to_model_actions] WARNING: Negative actions found!", actions)
    assert (actions >= 0).all(), f"[to_model_actions] Negative action index found: {actions}"
    if actions.ndim == 1:
        actions = actions.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif actions.ndim == 2:
        actions = actions.unsqueeze(0)  # [1, 1, T] if [1, T]
    assert actions.ndim == 3, f"Expected actions to have 3 dims, got {actions.shape}"
    actions = actions.to(torch.long)
    if device is not None:
        actions = actions.to(device)
    return actions


def to_numpy_cpu(tensor):
    """
    Move tensor to CPU and convert to numpy array.
    """
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return tensor
