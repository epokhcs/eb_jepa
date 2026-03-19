#!/bin/bash
# Script to run ATARI Breakout training with uv environment

set -e

echo "Activating uv environment..."
source .venv/bin/activate

echo "Setting PYTHONPATH..."
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH

echo "Disabling wandb..."
export WANDB_MODE=disabled

echo "Running ATARI Breakout training (1 epoch, small dataset)..."
python examples/ac_video_jepa/main.py \
  --fname examples/ac_video_jepa/cfgs/train_atari.yaml \
  --optim.epochs 1 \
  --data.size 1000 \
  --data.val_size 200 \
  --data.batch_size 32 \
  --data.num_workers 0

echo "Training complete!"
