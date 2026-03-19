#!/usr/bin/env python3
"""
Device detection test - run without PYTHONPATH to avoid module conflicts.
"""
import sys

# Clean Python path to avoid conflicts
sys.path = [p for p in sys.path if 'eb_jepa' not in p or '.venv' in p]

import torch


def main():
    print("🖥️  Device Information")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (NVIDIA GPU): {'✅ Available' if cuda_available else '❌ Not available'}")
    if cuda_available:
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Device count: {torch.cuda.device_count()}")

    # Check MPS
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon): {'✅ Available' if mps_available else '❌ Not available'}")
    if mps_available:
        print("  Device: Apple Silicon GPU (M-series)")

    # Best device
    print()
    if cuda_available:
        best = "cuda"
    elif mps_available:
        best = "mps"
    else:
        best = "cpu"

    print(f"Best available device: {best}")
    print("=" * 50)

    # Test the device
    print()
    print(f"Testing {best} device...")
    device = torch.device(best)
    x = torch.randn(100, 100).to(device)
    y = x @ x.T
    print(f"✅ Test passed! Tensor device: {y.device}")

    if best == "mps":
        print()
        print("🚀 MPS acceleration enabled!")
        print("Your Mac's Apple Silicon GPU will be used for training.")
        print("Expected speedup: ~5-10x faster than CPU")
    elif best == "cuda":
        print()
        print("🚀 CUDA acceleration enabled!")
        print("Your NVIDIA GPU will be used for training.")
    else:
        print()
        print("⚠️  Using CPU only")
        print("Training will be slower. Consider:")
        print("  - Mac: Run ./setup-mac.sh for MPS (GPU) support")
        print("  - Linux/Windows: Install CUDA-enabled PyTorch")


if __name__ == "__main__":
    main()
