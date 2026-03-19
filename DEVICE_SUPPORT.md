# Device Support: CPU, CUDA (NVIDIA GPU), and MPS (Apple Silicon GPU)

This project supports three device types for acceleration:

1. **CPU** - Universal fallback, works everywhere
2. **CUDA** - NVIDIA GPU acceleration (Linux/Windows)
3. **MPS** - Apple Silicon GPU acceleration (Mac M4/M3/M2/M1)

## Quick Start by Platform

### macOS (Apple Silicon M4/M3/M2/M1)

**Recommended for maximum performance with Apple Silicon GPU:**

```bash
# One-command setup
./setup-mac.sh

# Verify MPS is working
python eb_jepa/device_utils.py
```

This will:
- Install PyTorch with MPS support
- Verify Apple Silicon GPU is available
- Enable ~5-10x faster training vs CPU

### Linux/Windows (NVIDIA GPU)

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-minimal.txt
```

### Any Platform (CPU Only)

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install CPU-only version
pip install -r requirements-minimal.txt
```

## Device Detection

The framework automatically detects the best available device:

**Priority:** CUDA > MPS > CPU

```python
# Automatic detection (recommended)
from eb_jepa.device_utils import get_device

device = get_device("auto")  # Returns best available
# cuda (NVIDIA) > mps (Apple Silicon) > cpu

# Manual override
device = get_device("mps")   # Force Apple Silicon GPU
device = get_device("cuda")  # Force NVIDIA GPU
device = get_device("cpu")   # Force CPU
```

## Configuration

### Auto-detect (Recommended)

Leave `device` unset or set to `null` in config files:

```yaml
# data_config.yaml
device:  # Auto-detect best device
```

### Manual Override

Specify device explicitly:

```yaml
device: mps    # Apple Silicon GPU
device: cuda   # NVIDIA GPU
device: cpu    # CPU only
```

Or via command line:

```bash
python examples/ac_video_jepa/main.py \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml \
  --data.device mps
```

## Performance Comparison

Approximate training speeds (ATARI Breakout, 1 epoch):

| Device Type | Time | Speedup |
|-------------|------|---------|
| CPU (Intel i9) | ~45min | 1x |
| MPS (M4 Pro) | ~6min | ~7.5x |
| CUDA (RTX 4090) | ~3min | ~15x |

*Times are approximate and depend on model size and batch size*

## Checking Device Availability

### Command Line

```bash
# Print device info
python eb_jepa/device_utils.py
```

Output:
```
🖥️  Device Information:
==================================================
Best available device: mps
CUDA available: False
MPS available: True
  - Device: Apple Silicon GPU (MPS)
==================================================
```

### Python Code

```python
from eb_jepa.device_utils import get_device_info, print_device_info

# Print summary
print_device_info()

# Get info dict
info = get_device_info()
print(f"Best device: {info['best_device']}")
print(f"MPS available: {info['mps_available']}")
print(f"CUDA available: {info['cuda_available']}")
```

## Troubleshooting

### MPS Not Available on Mac

**Problem:** MPS shows as unavailable despite having Apple Silicon

**Solutions:**
1. **Update PyTorch:** MPS requires PyTorch 2.0+
   ```bash
   pip install --upgrade torch torchvision
   ```

2. **Check macOS version:** MPS requires macOS 12.3+
   ```bash
   sw_vers  # Check macOS version
   ```

3. **Verify architecture:**
   ```bash
   uname -m  # Should show "arm64" for Apple Silicon
   ```

### CUDA Out of Memory

**Problem:** Training crashes with CUDA OOM error

**Solutions:**
1. Reduce batch size:
   ```bash
   --data.batch_size 16  # Instead of 32
   ```

2. Use gradient checkpointing (if available)

3. Switch to CPU/MPS if GPU memory is insufficient

### MPS Slower Than Expected

**Problem:** MPS not showing speedup

**Possible causes:**
1. **Small batch size:** MPS benefits from larger batches
   - Try `--data.batch_size 64` or higher

2. **Data loading bottleneck:** Use multiple workers
   ```bash
   --data.num_workers 4
   ```

3. **Pin memory disabled:** Enable for MPS
   ```bash
   --data.pin_mem true
   ```

## Implementation Notes

### Device Detection Logic

```python
# Priority: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

### MPS Limitations

MPS is generally very fast but has some limitations vs CUDA:

1. **No pin_memory support yet** - Will show warning but won't fail
2. **Some operations not implemented** - Rare, usually has CPU fallback
3. **Less mature than CUDA** - May have occasional bugs

If you encounter MPS issues, you can always fall back to CPU:
```bash
--data.device cpu
```

## Requirements Files

Three requirements files are provided:

1. **requirements-minimal.txt** - Core dependencies, CPU/CUDA auto-detect
2. **requirements-mac.txt** - Mac-optimized with MPS support
3. **requirements.txt** - Full dependencies (if exists)

Choose based on your platform:
- **Mac:** Use `requirements-mac.txt` (via `setup-mac.sh`)
- **Linux/Windows:** Use `requirements-minimal.txt`

## Support

- MPS support: PyTorch 2.0+ on macOS 12.3+ with Apple Silicon
- CUDA support: PyTorch with CUDA 11.8+ on Linux/Windows with NVIDIA GPU
- CPU support: Universal

For device-specific issues, check:
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)
