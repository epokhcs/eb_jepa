#!/usr/bin/env python3
"""
Quick verification script for ATARI support implementation.

Tests basic functionality without running full training.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from eb_jepa.datasets.base import EnvBase, DatasetBase, NormalizerBase
        from eb_jepa.datasets.registry import EnvironmentRegistry
        from eb_jepa.datasets.atari import AtariEnv, AtariDataset, AtariConfig
        from eb_jepa.datasets.two_rooms import DotWall, WallDataset, WallDatasetConfig
        from eb_jepa.architectures import DiscreteActionEncoder, DiscreteInverseDynamicsModel
        from eb_jepa.losses import DiscreteInverseDynamicsLoss
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_registry():
    """Test environment registry."""
    print("\nTesting environment registry...")
    try:
        from eb_jepa.datasets.registry import EnvironmentRegistry

        envs = EnvironmentRegistry.list_environments()
        print(f"  Registered environments: {envs}")

        assert "two_rooms" in envs, "two_rooms not registered"
        assert "atari" in envs, "atari not registered"

        print("✓ Registry working correctly")
        return True
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        return False


def test_atari_env():
    """Test ATARI environment creation."""
    print("\nTesting ATARI environment...")
    try:
        from eb_jepa.datasets.atari import AtariEnv, AtariConfig

        config = AtariConfig(game_name="Breakout", device="cpu")
        env = AtariEnv(config)

        # Test action space info
        action_info = env.get_action_space_info()
        print(f"  Action space: {action_info}")
        assert action_info["type"] == "discrete"
        assert action_info["n"] == 4  # Breakout has 4 actions

        # Test observation space info
        obs_info = env.get_observation_space_info()
        print(f"  Observation space: {obs_info}")
        assert obs_info["shape"] == (1, 84, 84)

        # Test reset
        obs, info = env.reset()
        print(f"  Reset observation shape: {obs.shape}")
        assert obs.shape == (1, 84, 84)

        # Test step
        obs, reward, done, truncated, info = env.step(0)
        print(f"  Step observation shape: {obs.shape}")
        assert obs.shape == (1, 84, 84)

        env.close()
        print("✓ ATARI environment working correctly")
        return True
    except Exception as e:
        print(f"✗ ATARI environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_discrete_action_encoder():
    """Test discrete action encoder."""
    print("\nTesting discrete action encoder...")
    try:
        import torch
        from eb_jepa.architectures import DiscreteActionEncoder

        encoder = DiscreteActionEncoder(num_actions=4, embedding_dim=64)

        # Test forward pass
        actions = torch.tensor([[0], [1], [2], [3]]).unsqueeze(-1)  # [B, 1, 1]
        embedded = encoder(actions)

        print(f"  Input shape: {actions.shape}")
        print(f"  Output shape: {embedded.shape}")
        assert embedded.shape == (4, 64, 1)

        print("✓ Discrete action encoder working correctly")
        return True
    except Exception as e:
        print(f"✗ Discrete action encoder test failed: {e}")
        return False


def test_two_rooms_compatibility():
    """Test Two Rooms backward compatibility."""
    print("\nTesting Two Rooms backward compatibility...")
    try:
        from eb_jepa.datasets.two_rooms import DotWall, WallDatasetConfig
        from eb_jepa.datasets.registry import EnvironmentRegistry

        # Test direct instantiation (old way)
        config = WallDatasetConfig(device="cpu")
        env = DotWall(config)

        action_info = env.get_action_space_info()
        print(f"  Action space: {action_info}")
        assert action_info["type"] == "continuous"
        assert action_info["dim"] == 2

        # Test registry instantiation (new way)
        env2 = EnvironmentRegistry.create_env("two_rooms", config)
        action_info2 = env2.get_action_space_info()
        assert action_info == action_info2

        print("✓ Two Rooms backward compatibility maintained")
        return True
    except Exception as e:
        print(f"✗ Two Rooms compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("ATARI Support Verification")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Registry", test_registry()))
    results.append(("ATARI Environment", test_atari_env()))
    results.append(("Discrete Action Encoder", test_discrete_action_encoder()))
    results.append(("Two Rooms Compatibility", test_two_rooms_compatibility()))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
