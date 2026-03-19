#!/bin/bash
# Setup script for Mac with Apple Silicon (M4/M3/M2/M1) GPU acceleration

set -e

echo "🍎 Mac Apple Silicon Setup"
echo "============================"
echo ""

# Check if running on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is designed for macOS"
    echo "For Linux/Windows, use requirements-minimal.txt instead"
    exit 1
fi

# Check for Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "✅ Detected Apple Silicon (M-series chip)"
else
    echo "⚠️  Warning: Not running on Apple Silicon"
    echo "MPS acceleration may not be available"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if command -v uv &> /dev/null; then
    echo "Using uv (fast)..."
    uv venv .venv
else
    echo "Using python venv..."
    python3 -m venv .venv
fi

# Activate environment
echo ""
echo "Activating environment..."
source .venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv pip..."
    uv pip install -r requirements-mac.txt
else
    echo "Using pip..."
    pip install -r requirements-mac.txt
fi

# Verify MPS availability
echo ""
echo "Verifying MPS (Metal Performance Shaders) support..."
python3 << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
print(f"MPS built: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()}")

if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("\n✅ MPS is available! GPU acceleration enabled.")

    # Test MPS
    print("\nTesting MPS...")
    x = torch.randn(1000, 1000)
    x_mps = x.to('mps')
    y = x_mps @ x_mps.T
    print(f"Test tensor device: {y.device}")
    print("✅ MPS test passed!")
else:
    print("\n⚠️  MPS not available. Will use CPU.")
    print("Make sure you're running PyTorch 2.0+ on Apple Silicon")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================="
    echo "✅ Setup complete!"
    echo "=============================="
    echo ""
    echo "To activate the environment in the future:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "To run ATARI training with MPS acceleration:"
    echo "  ./run_atari.sh"
    echo ""
    echo "To check device info:"
    echo "  python eb_jepa/device_utils.py"
else
    echo ""
    echo "❌ Setup completed but MPS is not available"
    echo "Training will use CPU (slower)"
fi
