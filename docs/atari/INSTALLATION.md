# Installation Guide

Complete setup guide for EB-JEPA ATARI Breakout on Mac (MPS GPU) and Linux (CUDA GPU).

---

## Prerequisites

- **Python 3.12** (required)
- **Git** (for cloning the repository)
- **uv** or **pip** package manager
- **Hardware**: Mac with Apple Silicon (M4/M3/M2/M1) or Linux with NVIDIA GPU (recommended), or any CPU

---

## Quick Install by Platform

### Path 1: Mac with Apple Silicon (M4/M3/M2/M1) - Recommended

**One-command setup with MPS GPU acceleration (5-10x faster than CPU):**

```bash
./setup-mac.sh
```

This automated script:
- Creates optimized virtual environment
- Installs PyTorch with MPS support
- Installs all dependencies
- Verifies Apple Silicon GPU is working
- Shows expected performance

**Speedup:** ~5-10x faster than CPU

**Continue to [Environment Setup](#environment-setup) after running the script.**

---

### Path 2: Linux with NVIDIA GPU

**For CUDA acceleration (10-15x faster than CPU):**

```bash
# 1. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch with CUDA 11.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install project dependencies
pip install -r requirements-minimal.txt
```

**Alternative with uv (recommended):**

```bash
# 1. Create virtual environment with uv
uv venv .venv
source .venv/bin/activate

# 2. Install PyTorch with CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install project dependencies
uv pip install -r requirements-minimal.txt
```

**Speedup:** ~10-15x faster than CPU

---

### Path 3: Universal (CPU Only)

**Works on any platform (Mac, Linux, Windows):**

```bash
# Using uv (recommended)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-minimal.txt

# OR using standard pip
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
```

**Note:** CPU-only training is slower but works everywhere. Consider using a GPU for better performance.

---

### Path 4: Using Conda + uv

**If you prefer conda for environment management:**

```bash
# 1. Create conda environment with Python 3.12
conda create -n eb_jepa python=3.12 -y
conda activate eb_jepa

# 2. Install with uv (faster than conda/pip)
uv pip install -e . --group dev

# OR install with pip
pip install -e . --group dev
```

---

## Environment Setup

After installation, configure your environment:

### 1. Set PYTHONPATH

**Required for imports to work correctly:**

```bash
# Add to current shell session
export PYTHONPATH=$PWD:$PYTHONPATH

# OR for persistent configuration (recommended)
echo "export PYTHONPATH=$PWD:\$PYTHONPATH" >> ~/.bashrc  # Linux
echo "export PYTHONPATH=$PWD:\$PYTHONPATH" >> ~/.zshrc   # Mac
source ~/.bashrc  # or ~/.zshrc on Mac
```

### 2. Configure W&B (Optional)

**For experiment tracking with Weights & Biases:**

```bash
# Option A: Use your W&B account
export WANDB_API_KEY="your_api_key_here"

# Option B: Disable W&B logging
export WANDB_MODE=disabled

# Option C: Create .env file (recommended)
cat > .env <<EOF
WANDB_API_KEY=your_api_key_here
WANDB_MODE=online
EOF
```

**Note:** If you don't configure W&B, training will still work but won't log to W&B dashboard.

### 3. Configure Dataset Paths (Optional)

**For SLURM or custom dataset locations:**

```bash
export EBJEPA_DSETS=/path/to/datasets
export EBJEPA_CKPTS=/path/to/checkpoints  # Optional
```

**Default behavior:** Datasets and checkpoints are stored in the project directory if not specified.

---

## Verification

### Check Device Detection

Verify your GPU is detected correctly:

```bash
python eb_jepa/device_utils.py
```

**Expected output (Mac with M4):**
```
🖥️  Device Information
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Platform: Darwin
PyTorch Version: 2.10.0

Available Devices:
  ✅ MPS (Apple Silicon): Available
  ❌ CUDA (NVIDIA GPU): Not available
  ✅ CPU: Available

Selected Device: mps
Expected Speedup: ~5-10x vs CPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Expected output (Linux with NVIDIA GPU):**
```
Available Devices:
  ❌ MPS (Apple Silicon): Not available
  ✅ CUDA (NVIDIA GPU): Available
  ✅ CPU: Available

Selected Device: cuda
Expected Speedup: ~10-15x vs CPU
```

### Run Verification Test

Test that ATARI environment works:

```bash
python verify_atari_support.py
```

**Expected output:**
```
✓ PASS: Imports
✓ PASS: Registry
✓ PASS: ATARI Environment
✓ PASS: Discrete Action Encoder
✓ PASS: Two Rooms Compatibility

✓ All tests passed!
```

---

## Device Priority

The system automatically selects the best available device:

**Priority Order:** CUDA > MPS > CPU

- **CUDA**: Used on Linux/Windows with NVIDIA GPU
- **MPS**: Used on Mac with Apple Silicon (M4/M3/M2/M1)
- **CPU**: Fallback for all platforms

**Manual Override** (if needed):

```bash
# Force specific device
python examples/ac_video_jepa/main.py \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml \
  --data.device mps    # or cuda, or cpu
```

---

## Performance Comparison

### Training Speed (1 epoch, 10K samples)

| Device | Time | Speedup | Availability |
|--------|------|---------|--------------|
| **CPU** | ~45 min | 1x | ✅ All platforms |
| **MPS (M4)** | ~6 min | ~7.5x | ✅ Mac M4/M3/M2/M1 |
| **CUDA (RTX 4090)** | ~3 min | ~15x | ✅ Linux with NVIDIA |

### Batch Size Recommendations

| Device | Batch Size | Memory Usage |
|--------|-----------|--------------|
| **CPU** | 16 | ~4GB RAM |
| **MPS (M4 16GB)** | 32 | ~8GB unified memory |
| **CUDA (24GB)** | 64 | ~12GB VRAM |

**Note:** Adjust batch size in config file: `data.batch_size: 32`

---

## Dependencies

### Core Dependencies

Installed automatically via `requirements-minimal.txt`:

- **PyTorch** - Deep learning framework
- **Gymnasium** - RL environment interface
- **ALE-Py** - Arcade Learning Environment (ATARI games)
- **Fire** - CLI tool
- **OmegaConf** - Configuration management
- **Einops** - Tensor operations
- **NumPy, Pandas** - Numerical computing
- **Matplotlib, Seaborn** - Visualization
- **OpenCV** - Image processing
- **TQDM** - Progress bars
- **Weights & Biases** - Experiment tracking (optional)

### Development Dependencies (Optional)

For contributing or development:

```bash
uv pip install -e . --group dev
```

Includes: pytest, black, isort, autoflake

---

## Troubleshooting Installation

### ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'eb_jepa'`

**Solution:**
```bash
# Check PYTHONPATH
echo $PYTHONPATH  # Should include project root

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

### ALE namespace not found

**Problem:** `gymnasium.error.NamespaceNotFound: Namespace ALE not found`

**Solution:**
```bash
# ALE-Py should auto-register, but if not:
python -c "import ale_py; import gymnasium as gym; gym.register_envs(ale_py)"
```

### MPS not detected on Mac

**Problem:** MPS shows as "Not available" on Mac with Apple Silicon

**Solutions:**
1. **Update PyTorch:** Ensure you have PyTorch 2.0+
   ```bash
   pip install --upgrade torch torchvision
   ```

2. **Check macOS version:** MPS requires macOS 12.3+
   ```bash
   sw_vers  # Should show 12.3 or higher
   ```

3. **Verify Apple Silicon:** Ensure you're on M4/M3/M2/M1
   ```bash
   sysctl -n machdep.cpu.brand_string
   ```

### CUDA not detected on Linux

**Problem:** CUDA shows as "Not available" with NVIDIA GPU

**Solutions:**
1. **Install NVIDIA drivers:**
   ```bash
   nvidia-smi  # Should show GPU info
   ```

2. **Install correct PyTorch:**
   ```bash
   # Check CUDA version
   nvcc --version

   # Install matching PyTorch (example for CUDA 11.8)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Import errors after installation

**Problem:** Various import errors

**Solution:**
```bash
# Reinstall in development mode
pip install -e .

# Verify installation
pip list | grep torch
pip list | grep gymnasium
```

---

## Next Steps

✅ Installation complete! Now you can:

1. **[Quickstart](QUICKSTART.md)** - Train your first model in 5 minutes
2. **[Training Guide](TRAINING.md)** - Full configuration options
3. **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

---

## Installation Summary

### Minimal Installation (3 commands)

**Mac:**
```bash
./setup-mac.sh
export PYTHONPATH=$PWD:$PYTHONPATH
python verify_atari_support.py
```

**Linux with GPU:**
```bash
uv venv .venv && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements-minimal.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Any Platform (CPU):**
```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements-minimal.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

Ready to train? → **[Quickstart Guide](QUICKSTART.md)**

---

**Last Updated:** March 21, 2026
