"""
Environment registry for dynamic loading of environments and datasets.

This registry pattern allows environments to be registered and created
dynamically without hardcoded imports in the core framework.
"""

from typing import Dict, Callable, Type, Any
from .base import EnvBase, DatasetBase, DatasetConfigBase


class EnvironmentRegistry:
    """Registry for environments and their associated datasets."""

    _envs: Dict[str, Type[EnvBase]] = {}
    _datasets: Dict[str, Type[DatasetBase]] = {}
    _configs: Dict[str, Type[DatasetConfigBase]] = {}
    _env_creators: Dict[str, Callable] = {}

    @classmethod
    def register_env(
        cls,
        name: str,
        env_class: Type[EnvBase],
        dataset_class: Type[DatasetBase],
        config_class: Type[DatasetConfigBase],
        env_creator: Callable = None,
    ):
        """
        Register a new environment.

        Args:
            name: Environment name (e.g., "two_rooms", "atari")
            env_class: Environment class implementing EnvBase
            dataset_class: Dataset class implementing DatasetBase
            config_class: Config class extending DatasetConfigBase
            env_creator: Optional factory function for creating env instances
        """
        cls._envs[name] = env_class
        cls._datasets[name] = dataset_class
        cls._configs[name] = config_class
        if env_creator:
            cls._env_creators[name] = env_creator

    @classmethod
    def create_env(cls, name: str, config: DatasetConfigBase, **kwargs) -> EnvBase:
        """
        Create an environment instance.

        Args:
            name: Environment name
            config: Configuration object
            **kwargs: Additional environment-specific parameters

        Returns:
            Environment instance

        Raises:
            ValueError: If environment name is not registered
        """
        if name not in cls._envs:
            available = ", ".join(cls.list_environments())
            raise ValueError(
                f"Unknown environment: {name}. "
                f"Available environments: {available}"
            )

        if name in cls._env_creators:
            return cls._env_creators[name](config=config, **kwargs)
        else:
            return cls._envs[name](config=config, **kwargs)

    @classmethod
    def create_dataset(cls, name: str, config: DatasetConfigBase) -> DatasetBase:
        """
        Create a dataset instance.

        Args:
            name: Environment name
            config: Configuration object

        Returns:
            Dataset instance

        Raises:
            ValueError: If environment name is not registered
        """
        if name not in cls._datasets:
            available = ", ".join(cls.list_environments())
            raise ValueError(
                f"Unknown environment: {name}. "
                f"Available environments: {available}"
            )
        return cls._datasets[name](config=config)

    @classmethod
    def get_config_class(cls, name: str) -> Type[DatasetConfigBase]:
        """
        Get the config class for an environment.

        Args:
            name: Environment name

        Returns:
            Config class

        Raises:
            ValueError: If environment name is not registered
        """
        if name not in cls._configs:
            available = ", ".join(cls.list_environments())
            raise ValueError(
                f"Unknown environment: {name}. "
                f"Available environments: {available}"
            )
        return cls._configs[name]

    @classmethod
    def list_environments(cls) -> list:
        """
        List all registered environments.

        Returns:
            List of environment names
        """
        return list(cls._envs.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if an environment is registered.

        Args:
            name: Environment name

        Returns:
            True if registered, False otherwise
        """
        return name in cls._envs


def register_environment(name: str):
    """
    Decorator for registering environments.

    Example:
        @register_environment("my_env")
        class MyEnv(EnvBase):
            pass
    """

    def decorator(env_class: Type[EnvBase]):
        # This decorator is a convenience wrapper
        # Actual registration should be done explicitly with dataset and config
        return env_class

    return decorator
