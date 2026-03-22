# Troubleshooting Guide

Common issues and solutions (SSL content removed per user request).

## Installation Issues

### ModuleNotFoundError
**Solution:** Set PYTHONPATH
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### ALE namespace not found
**Solution:** Verify ale-py installation
```bash
pip install ale-py gymnasium
python -c "import ale_py; import gymnasium as gym; gym.register_envs(ale_py)"
```

## Training Issues

### CUDA/MPS Out of Memory
**Solution:** Reduce batch_size in config
```yaml
data:
  batch_size: 16  # Reduce from 32
```

### Training Too Slow
**Check device detection:**
```bash
python eb_jepa/device_utils.py
```

### UnboundLocalError
**Status:** Fixed in commit 30838ce
**Solution:** Update to latest code

## Evaluation Issues

### Planning Evaluation Crashes
**Status:** Fixed in current version  
**Solution:** Update code, ensure using latest checkpoint

### Low Planning Scores
**Not a bug:** May need hyperparameter tuning
**See:** EVALUATION.md for expected ranges

## Performance Issues

### MPS Slower Than Expected
**Common causes:**
- Batch size too small (increase to 32)
- Data loading bottleneck (check num_workers)

### High Prediction MSE
**Common causes:**
- Insufficient training epochs (try 50 instead of 5)
- Improper regularization coefficients

## Environment Issues

### W&B Not Logging
**Solution:** Set environment variable
```bash
export WANDB_MODE=disabled
# or configure API key
export WANDB_API_KEY=your_key
```

## Getting More Help

- GitHub Issues: https://github.com/facebookresearch/eb_jepa/issues
- [INSTALLATION.md](INSTALLATION.md) for setup issues
- [ARCHITECTURE.md](ARCHITECTURE.md) for design questions

**Last Updated:** March 21, 2026
