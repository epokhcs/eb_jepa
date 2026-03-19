#!/bin/bash
# Simple script to run Two Rooms training with uv environment

set -e

echo "Activating uv environment..."
source .venv/bin/activate

echo "Setting PYTHONPATH..."
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH

echo "Disabling wandb..."
export WANDB_MODE=disabled

echo "Running Two Rooms training (1 epoch, small dataset)..."
python examples/ac_video_jepa/main.py \
  --fname examples/ac_video_jepa/cfgs/train.yaml \
  --optim.epochs 1 \
  --data.size 100 \
  --data.val_size 20 \
  --data.num_workers 0

echo "Training complete!"
