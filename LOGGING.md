# Logging Infrastructure

The training pipeline now includes comprehensive metrics logging and automatic wandb integration.

## Features

### 1. Local Metrics Logging

All training metrics are automatically saved to `logs/training/` directory:

- **JSON format**: `training_YYYYMMDD_HHMMSS.json` - Complete metrics history
- **CSV format**: `training_YYYYMMDD_HHMMSS.csv` - Spreadsheet-friendly format
- **Latest symlinks**: `training_latest.json` and `training_latest.csv` - Always point to current run

### 2. Tracked Metrics

**Per-iteration metrics** (logged every N steps):
- `total_loss`: Combined JEPA + probe loss
- `reg_loss`: Regularization loss (VIC-Reg components)
- `pred_loss`: Prediction loss
- `probe_loss`: XY position prediction loss (Two Rooms only)
- `jepa_lr`: JEPA optimizer learning rate
- `probe_lr`: Probe optimizer learning rate
- `itr_time`: Iteration time in seconds

**Epoch summary metrics**:
- `epoch_time_seconds`: Total epoch duration
- Averaged losses from the epoch

### 3. Automatic W&B Integration

Wandb logging is now controlled via `.env` file:

1. **Create `.env` file** (from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. **Add your API key**:
   ```bash
   # In .env file:
   WANDB_API_KEY=your_key_here
   ```

3. **Configure SSL verification** (if needed):
   ```bash
   # In .env file (optional):
   # Set to false if you encounter SSL certificate errors
   WANDB_VERIFY_SSL=false
   ```

4. **That's it!** Wandb will automatically enable if the key is present.

4. **To disable wandb** (even with API key):
   ```bash
   # In .env file:
   WANDB_MODE=disabled
   ```

### SSL Certificate Issues

If you encounter SSL certificate verification errors with W&B (common on corporate networks or with certain security configurations):

```bash
# In .env file:
WANDB_VERIFY_SSL=false
```

**Security Note**: Disabling SSL verification reduces security. Only use this if you trust your network and cannot resolve certificate issues through proper certificate installation.

## Usage

### Training with Logging

```bash
# Normal training - metrics auto-logged to logs/training/
./run_atari.sh
```

All metrics are saved automatically:
- Local JSON/CSV files are created in `logs/training/{experiment_name}/`
- If `WANDB_API_KEY` is in `.env`, metrics also go to W&B

### Viewing Metrics

**Local files**:
```bash
# View latest JSON metrics
cat logs/training/{exp_name}/training_latest.json

# Open CSV in spreadsheet
open logs/training/{exp_name}/training_latest.csv

# View experiment config
cat logs/training/{exp_name}/config.json

# View summary statistics
cat logs/training/{exp_name}/training_summary.json
```

**W&B dashboard**:
```bash
# If wandb is enabled, view at:
# https://wandb.ai/{your_username}/eb_jepa
```

### Analyzing Metrics

```python
import pandas as pd
import json

# Load metrics
df = pd.read_csv("logs/training/{exp_name}/training_latest.csv")

# Plot loss curves
df[['epoch', 'total_loss', 'reg_loss', 'pred_loss']].plot(x='epoch')

# Load config
with open("logs/training/{exp_name}/config.json") as f:
    config = json.load(f)
    print(config['model'])
```

## Log Directory Structure

```
logs/
├── training/
│   └── {experiment_name}/
│       ├── config.json                    # Experiment configuration
│       ├── training_20260319_143022.json  # Timestamped metrics
│       ├── training_20260319_143022.csv
│       ├── training_latest.json           # Symlink to latest
│       ├── training_latest.csv            # Symlink to latest
│       └── training_summary.json          # Summary statistics
└── inference/                             # Reserved for future inference logging
    └── {experiment_name}/
```

## Files Modified

1. **[eb_jepa/metrics_logger.py](eb_jepa/metrics_logger.py)** (NEW)
   - `MetricsLogger` class for JSON/CSV logging
   - `setup_wandb_from_env()` for automatic wandb detection

2. **[examples/ac_video_jepa/main.py](examples/ac_video_jepa/main.py)**
   - Added metrics logger initialization
   - Log metrics every iteration
   - Log epoch summaries
   - Save config to logs

3. **[.env](.env)** (NEW, gitignored)
   - Contains your WANDB_API_KEY

4. **[.env.example](.env.example)** (NEW)
   - Template for creating `.env` file

5. **[requirements-minimal.txt](requirements-minimal.txt)**
   - Added `python-dotenv>=1.0.0`

6. **[requirements-mac.txt](requirements-mac.txt)**
   - Added `python-dotenv>=1.0.0`

## Security Note

⚠️ **Never commit `.env` to git!** It contains your API keys.

The `.env` file is already in `.gitignore` to prevent accidental commits. Always use `.env.example` as a template.

## Future: Inference Logging

The infrastructure also supports inference logging (currently unused):

```python
from eb_jepa.metrics_logger import MetricsLogger

# For inference/evaluation
metrics_logger = MetricsLogger(
    log_dir=Path("logs/inference"),
    experiment_name="eval_breakout",
    mode="inference",
)

# Log inference metrics
metrics_logger.log_metrics(
    epoch=0,
    step=episode_num,
    metrics={
        "episode_reward": total_reward,
        "episode_length": episode_length,
        "success_rate": success_rate,
    },
)
```

## Troubleshooting

### Issue: W&B authentication errors or SSL certificate failures

**Symptom**:
```
wandb.errors.errors.CommError: Error uploading run: returned error 401
SSLError(SSLCertVerificationError(...))
```

**Solution**: Disable SSL certificate verification

1. Edit your `.env` file:
   ```bash
   WANDB_VERIFY_SSL=false
   ```

2. Restart training - SSL verification will be automatically disabled

**Why this happens**: Common in corporate environments with proxy/firewall settings, or when system certificates are not properly configured.

**Security Note**: Only disable SSL verification if you trust your network. For production environments, properly configure SSL certificates instead.

---

**Issue**: Metrics not saving

**Solution**: Check that `logs/training/` directory exists (created automatically)

---

**Issue**: Wandb not auto-enabling

**Solution**:
1. Verify `.env` file exists: `ls -la .env`
2. Check API key is set: `cat .env`
3. Make sure python-dotenv is installed: `pip list | grep dotenv`

---

**Issue**: "Permission denied" on log files

**Solution**: Check write permissions: `chmod -R u+w logs/`

## Summary

✅ **Automatic logging** - No configuration needed, just train!
✅ **Local storage** - JSON/CSV files for offline analysis
✅ **W&B integration** - Auto-enabled if API key in `.env`
✅ **Secure** - `.env` gitignored to protect API keys
✅ **Future-ready** - Infrastructure for inference logging ready
