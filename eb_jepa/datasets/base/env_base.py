from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import torch
import numpy as np


class EnvBase(ABC):
    """
    Abstract base class for all environments in the ac_video_jepa framework.

    This interface handles the unified API for:
    - Different action spaces (continuous vs discrete)
    - Different observation shapes and types
    - Different evaluation metrics (position-based vs score-based)
    - Different rendering approaches
    """

    @abstractmethod
    def reset(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            **kwargs: Environment-specific reset parameters

        Returns:
            obs: Observation tensor [C, H, W]
            info: Dictionary with environment-specific info
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Action (continuous or discrete depending on action_space)

        Returns:
            obs: Observation tensor [C, H, W]
            reward: Scalar reward
            done: Episode terminated
            truncated: Episode truncated
            info: Dictionary with environment-specific info
        """
        pass

    @abstractmethod
    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get action space information.

        Returns:
            Dictionary with keys:
                'type': 'continuous' or 'discrete'
                'dim': int (action dimension)
                'low': Optional[float] (for continuous)
                'high': Optional[float] (for continuous)
                'n': Optional[int] (for discrete - number of actions)
        """
        pass

    @abstractmethod
    def get_observation_space_info(self) -> Dict[str, Any]:
        """
        Get observation space information.

        Returns:
            Dictionary with keys:
                'shape': Tuple[int, int, int]  # (C, H, W)
                'dtype': str
        """
        pass

    @abstractmethod
    def eval_state(self, goal_state: Any, curr_state: Any, **kwargs) -> Dict[str, Any]:
        """
        Evaluate current state against goal.

        Args:
            goal_state: Goal state (format depends on environment)
            curr_state: Current state (format depends on environment)
            **kwargs: Environment-specific parameters

        Returns:
            Dictionary with keys:
                'success': bool
                'metric': float (distance, score difference, etc.)
        """
        pass

    @abstractmethod
    def render_from_latent(self, latent: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Render pixel observations from latent representations.

        This is used for visualization during planning.

        Args:
            latent: Latent representation from the model
            **kwargs: Environment-specific parameters

        Returns:
            frames: Tensor of rendered frames
        """
        pass

    @property
    @abstractmethod
    def normalizer(self):
        """Return the normalizer for this environment."""
        pass

    def get_goal_from_info(self, info: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """
        Extract goal observation and goal state from info dict.

        Args:
            info: Info dictionary returned by reset() or step()

        Returns:
            goal_obs: Goal observation tensor
            goal_state: Goal state representation (env-specific)
        """
        if "target_obs" in info and "target_position" in info:
            return info["target_obs"], info["target_position"]
        elif "target_obs" in info:
            return info["target_obs"], None
        else:
            raise NotImplementedError(
                f"Environment must provide goal in info dict. "
                f"Available keys: {list(info.keys())}"
            )
