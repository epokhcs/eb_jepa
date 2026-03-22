from typing import Any, Dict, Tuple, Optional
import numpy as np
import torch

try:
    import gymnasium as gym
    import ale_py
    # Register ALE environments with gymnasium
    gym.register_envs(ale_py)
except ImportError:
    raise ImportError(
        "gymnasium and ale-py are required for ATARI environments. "
        "Install with: pip install gymnasium ale-py"
    )

from ..base import EnvBase
from .config import AtariConfig
from .normalizer import AtariNormalizer


class AtariEnv(EnvBase):
    """
    ATARI environment implementing EnvBase interface.

    Wraps gymnasium ATARI environments with preprocessing for ac_video_jepa.
    """

    def __init__(
        self,
        config: AtariConfig,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ATARI environment.

        Args:
            config: AtariConfig with game settings
            render_mode: Rendering mode for gymnasium
            **kwargs: Additional environment arguments (Two Rooms specific args are filtered out)
        """
        self.config = config

        # Filter out Two Rooms specific kwargs that gymnasium doesn't understand
        atari_kwargs = {}
        for key, value in kwargs.items():
            if key not in ['n_allowed_steps', 'level', 'wall_x', 'door_y']:
                atari_kwargs[key] = value

        # Create base ATARI environment
        try:
            self.env = gym.make(
                f"ALE/{config.game_name}-v5",
                frameskip=config.frame_skip,
                render_mode=render_mode,
                **atari_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create ATARI environment '{config.game_name}'. "
                f"Make sure ale-py is installed: pip install ale-py\n"
                f"Error: {e}"
            )

        # Setup device with MPS support for Apple Silicon
        if config.device is None:
            # Auto-detect: CUDA > MPS > CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        # Initialize normalizer
        if config.normalize:
            self._normalizer = AtariNormalizer()
        else:
            self._normalizer = None

        # Frame preprocessing settings
        self.grayscale = config.grayscale
        self.target_size = (config.img_size, config.img_size)

    def reset(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset the ATARI environment.

        Returns:
            obs: Observation tensor [C, H, W]
            info: Info dictionary
        """
        obs, info = self.env.reset(**kwargs)
        obs_tensor = self._preprocess_observation(obs)
        return obs_tensor, info

    def step(self, action: np.ndarray) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        """
        Take a step in the ATARI environment.

        Args:
            action: Discrete action index (scalar or [1] array)

        Returns:
            obs: Observation tensor [C, H, W]
            reward: Scalar reward
            done: Episode terminated
            truncated: Episode truncated
            info: Info dictionary
        """
        # Convert action to integer
        if isinstance(action, (np.ndarray, torch.Tensor)):
            action_idx = int(action.flatten()[0])
        else:
            action_idx = int(action)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action_idx)

        # Preprocess observation
        obs_tensor = self._preprocess_observation(obs)

        return obs_tensor, float(reward), terminated, truncated, info

    def _preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
        """
        Preprocess ATARI observation.

        Args:
            obs: Raw ATARI observation [H, W, C] or [H, W]

        Returns:
            Preprocessed observation [C, H, W]
        """
        # Convert to tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)

        # Convert to float
        obs = obs.float()

        # Handle grayscale conversion
        if self.grayscale:
            if obs.ndim == 3 and obs.shape[-1] == 3:
                # RGB to grayscale: use standard weights
                obs = 0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
            elif obs.ndim == 2:
                # Already grayscale
                pass
            else:
                raise ValueError(f"Unexpected observation shape: {obs.shape}")

            # Add channel dimension: [H, W] -> [1, H, W]
            if obs.ndim == 2:
                obs = obs.unsqueeze(0)
        else:
            # Keep RGB: [H, W, 3] -> [3, H, W]
            if obs.ndim == 3:
                obs = obs.permute(2, 0, 1)

        # Resize if needed
        if obs.shape[-2:] != self.target_size:
            import torch.nn.functional as F
            obs = F.interpolate(
                obs.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Normalize pixels [0, 255] -> [0, 1]
        if self._normalizer is not None:
            obs = self._normalizer.normalize_state(obs)

        # Move to device
        obs = obs.to(self.device)

        return obs

    @property
    def action_space(self):
        """Expose gymnasium action space for planning compatibility."""
        return self.env.action_space

    @property
    def observation_space(self):
        """Expose gymnasium observation space for compatibility."""
        return self.env.observation_space

    def get_action_space_info(self) -> Dict[str, Any]:
        """Return action space information for ATARI."""
        return {
            "type": "discrete",
            "n": self.env.action_space.n,
            "dim": 1,  # Single discrete action
        }

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Return observation space information for ATARI."""
        n_channels = 1 if self.grayscale else 3
        return {
            "shape": (n_channels, self.target_size[0], self.target_size[1]),
            "dtype": "float32",
        }

    def eval_state(
        self,
        goal_state: Any,
        curr_state: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate current state against goal.

        For ATARI, evaluation is typically score-based rather than
        state-based (unlike Two Rooms).

        Args:
            goal_state: Goal score or state
            curr_state: Current score or state
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'success' and 'metric'
        """
        if isinstance(goal_state, (int, float)) and isinstance(curr_state, (int, float)):
            # Score-based evaluation
            score_diff = abs(goal_state - curr_state)
            success = score_diff < self.config.success_score_threshold
            return {
                "success": success,
                "metric": score_diff,
            }
        else:
            # State-based evaluation (not typically used for ATARI)
            # Could use frame similarity metrics like MSE
            raise NotImplementedError(
                "State-based evaluation not implemented for ATARI. "
                "Use score-based evaluation instead."
            )

    def render_from_latent(self, latent: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Render pixel observations from latent representations.

        For ATARI, this requires a decoder network which should be
        provided through the planning adapter.

        Args:
            latent: Latent representation
            **kwargs: Additional parameters

        Returns:
            Rendered frames
        """
        raise NotImplementedError(
            "ATARI rendering from latent requires a decoder network. "
            "Use AtariPlanningAdapter with a trained decoder."
        )

    @property
    def normalizer(self):
        """Return the normalizer for this environment."""
        return self._normalizer

    def close(self):
        """Close the environment."""
        if hasattr(self, 'env'):
            self.env.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
