"""
Device detection utilities for CPU, CUDA (NVIDIA GPU), and MPS (Apple Silicon GPU).
"""
from typing import Literal

# Import torch after type definitions to avoid circular imports
try:
    import torch
except ImportError:
    print("PyTorch not installed. Please run: pip install torch")
    exit(1)

DeviceType = Literal["cpu", "cuda", "mps", "auto"]


def get_available_device() -> str:
    """
    Detect the best available device.

    Priority:
    1. CUDA (NVIDIA GPU) if available
    2. MPS (Apple Silicon GPU) if available
    3. CPU as fallback

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device(device: DeviceType = "auto") -> torch.device:
    """
    Get PyTorch device with auto-detection support.

    Args:
        device: Device type - "cpu", "cuda", "mps", or "auto"
                "auto" will detect the best available device

    Returns:
        torch.device object

    Examples:
        >>> device = get_device("auto")  # Auto-detect
        >>> device = get_device("mps")   # Force Apple Silicon GPU
        >>> device = get_device("cpu")   # Force CPU
    """
    if device == "auto":
        device = get_available_device()

    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("⚠️  MPS not available, falling back to CPU")
            device = "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = "cpu"

    return torch.device(device)


def get_device_info() -> dict:
    """
    Get information about available devices.

    Returns:
        Dict with device availability and names
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "best_device": get_available_device(),
    }

    if info["cuda_available"]:
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()

    if info["mps_available"]:
        info["mps_device_name"] = "Apple Silicon GPU (MPS)"

    return info


def print_device_info():
    """Print available device information."""
    info = get_device_info()

    print("🖥️  Device Information:")
    print("=" * 50)
    print(f"Best available device: {info['best_device']}")
    print(f"CUDA available: {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"  - Device: {info['cuda_device_name']}")
        print(f"  - Count: {info['cuda_device_count']}")

    print(f"MPS available: {info['mps_available']}")
    if info["mps_available"]:
        print(f"  - Device: {info['mps_device_name']}")

    print("=" * 50)


if __name__ == "__main__":
    print_device_info()

    # Test device detection
    device = get_device("auto")
    print(f"\nSelected device: {device}")

    # Test tensor on device
    x = torch.randn(10, 10).to(device)
    print(f"Tensor device: {x.device}")
