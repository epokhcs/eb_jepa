# Running ac_video_jepa with ATARI Support

## Quick Start Guide

This guide shows how to run the ac_video_jepa framework with both Two Rooms (original) and ATARI Breakout (new).

## Prerequisites

- Python 3.12
- `uv` package manager (or pip)
- Mac/Linux environment

## Setup (One-time)

### 1. Create Virtual Environment

#### Option A: Mac with Apple Silicon (M4/M3/M2/M1) - Recommended

```bash
# One-command setup with MPS (GPU) acceleration
./setup-mac.sh
```

This automatically:
- Creates `.venv` environment
- Installs Mac-optimized dependencies
- Verifies Apple Silicon GPU (MPS) support
- Enables ~5-10x faster training vs CPU

#### Option B: Standard Setup (Any Platform)

```bash
# Using uv (recommended)
uv venv .venv

# Or using venv
python -m venv .venv
```

### 2. Install Dependencies

#### Mac (after running setup-mac.sh)
Already done! Skip to step 3.

#### Other Platforms

```bash
# Activate environment
source .venv/bin/activate

# Install minimal requirements
uv pip install -r requirements-minimal.txt

# Or with pip:
# pip install torch torchvision einops gymnasium ale-py fire omegaconf pyyaml scipy numpy pandas matplotlib opencv-python tqdm wandb scikit-learn seaborn imageio imageio-ffmpeg
```

### 3. Set Environment Variables

```bash
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH
export WANDB_MODE=disabled  # Disable wandb for testing
```

## Running Two Rooms

### Option 1: Using the script

```bash
./run_two_rooms.sh
```

### Option 2: Manual command

```bash
source .venv/bin/activate
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH
export WANDB_MODE=disabled

python examples/ac_video_jepa/main.py \
  --fname examples/ac_video_jepa/cfgs/train.yaml \
  --optim.epochs 1 \
  --data.size 1000 \
  --data.val_size 200 \
  --data.batch_size 32 \
  --data.num_workers 0
```

## Running ATARI Breakout

### Option 1: Quick test

```bash
source .venv/bin/activate
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH
export WANDB_MODE=disabled

python examples/ac_video_jepa/main.py \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml \
  --optim.epochs 1 \
  --data.size 1000 \
  --data.val_size 200 \
  --data.batch_size 32 \
  --data.num_workers 0
```

### Option 2: Full training

```bash
source .venv/bin/activate
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH

python examples/ac_video_jepa/main.py \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml
```

## Running Verification Tests

```bash
source .venv/bin/activate
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH

python verify_atari_support.py
```

Expected output:
```
✓ PASS: Imports
✓ PASS: Registry
✓ PASS: ATARI Environment
✓ PASS: Discrete Action Encoder
✓ PASS: Two Rooms Compatibility

✓ All tests passed!
```

## Configuration Options

### Key Parameters

- `--optim.epochs N` - Number of training epochs
- `--data.size N` - Training dataset size
- `--data.val_size N` - Validation dataset size
- `--data.num_workers N` - Number of data loading workers (use 0 for debugging)
- `--data.batch_size N` - Batch size
- `--logging.log_wandb false` - Disable wandb logging

### Environment-specific Settings

**Two Rooms:**
- Environment: `data.env_name: two_rooms`
- Action space: Continuous 2D
- Observation: 2-channel (65×65)

**ATARI Breakout:**
- Environment: `data.env_name: atari`
- Game: `data.game_name: Breakout`
- Action space: Discrete (4 actions)
- Observation: 1-channel grayscale (84×84)

## Troubleshooting

### ModuleNotFoundError

If you get `ModuleNotFoundError`, make sure:
1. Virtual environment is activated: `source .venv/bin/activate`
2. PYTHONPATH is set: `export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH`
3. Dependencies are installed: `uv pip install -r requirements-minimal.txt`

### "No module named 'examples'"

Don't use `python -m examples.ac_video_jepa.main`. Instead:
- Set PYTHONPATH first
- Run directly: `python examples/ac_video_jepa/main.py`

### ALE namespace not found

Make sure ale-py is installed and registered:
```python
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
```

This is done automatically when importing the ATARI environment.

### Wandb errors

Disable wandb with:
```bash
export WANDB_MODE=disabled
```

Or use your API key:
```bash
export WANDB_API_KEY="your_key_here"
```

## Device Support

The framework automatically detects the best available device:
- **CUDA** (NVIDIA GPU) if available
- **MPS** (Apple Silicon M4/M3/M2/M1 GPU) if on Mac
- **CPU** as fallback

To check your device:
```bash
python eb_jepa/device_utils.py
```

For Mac users with Apple Silicon, use `./setup-mac.sh` for automatic MPS (GPU) acceleration setup.

See [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md) for detailed device configuration.

### UnboundLocalError in training

This was a bug in the original codebase when running with very small datasets (batch_size > dataset_size). Fixed in commit 30838ce. Make sure your batch_size is smaller than your dataset size.

## Project Structure

```
eb_jepa/
├── eb_jepa/
│   ├── datasets/
│   │   ├── base/           # Abstract interfaces
│   │   ├── two_rooms/      # Original environment
│   │   ├── atari/          # ATARI implementation
│   │   ├── registry.py     # Environment registry
│   │   └── utils.py        # Data loading utilities
│   ├── architectures.py    # Model architectures
│   ├── losses.py           # Loss functions
│   ├── planning.py         # Planning algorithms
│   └── planning_adapters.py # Environment-specific adapters
├── examples/
│   └── ac_video_jepa/
│       ├── cfgs/
│       │   ├── train.yaml        # Two Rooms config
│       │   └── train_atari.yaml  # ATARI config
│       ├── main.py               # Training script
│       └── ATARI_README.md       # Detailed ATARI docs
├── verify_atari_support.py       # Verification script
├── run_two_rooms.sh              # Convenience script
└── requirements-minimal.txt      # Dependencies
```

## What's Working

✅ **Environment abstraction layer** - Plugin architecture for multiple environments
✅ **Two Rooms** - Original environment, backward compatible
✅ **ATARI Breakout** - New implementation with discrete actions
✅ **Discrete action support** - Embeddings, IDM, planning
✅ **Verification tests** - All 5 tests passing
✅ **Training infrastructure** - Loads configs, creates models, starts training

## Known Issues

- Wandb API key authentication may fail (use `WANDB_MODE=disabled`)

## Next Steps

1. **Test training** - Run the scripts above with small datasets
2. **Full training** - Remove epoch/size limits for real experiments
3. **Evaluation** - Add `--meta.eval_only_mode true` to run planning evaluation
4. **Custom games** - Change `data.game_name` to try other ATARI games

## Support

- See [ATARI_README.md](examples/ac_video_jepa/ATARI_README.md) for detailed documentation
- See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details
- Run `python verify_atari_support.py` to check your setup
