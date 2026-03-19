#!/bin/bash
# Generate ATARI Breakout pretraining dataset

set -e

echo "🎮 ATARI Dataset Generation"
echo "============================"

# Activate environment
source .venv/bin/activate
export PYTHONPATH=/Users/pdiprodi/DevOps/github/eb_jepa:$PYTHONPATH

# Default: 1000 trajectories for quick testing
# For full dataset, use 10000+
python generate_atari_dataset.py \
  --game Breakout \
  --num_trajectories 1000 \
  --num_steps 200 \
  --output_dir data/atari \
  --split train

echo ""
echo "✅ Training dataset created!"
echo ""
echo "To generate validation set:"
echo "./generate_atari_dataset.py --game Breakout --num_trajectories 200 --split val"
