#!/bin/bash
# Quick standalone verification of human gameplay data

echo "=========================================="
echo "HUMAN GAMEPLAY DATA VERIFICATION"
echo "=========================================="
echo ""

# Check if dataset exists
if [ ! -f "data/systematic_sweep/breakout_human_gameplay.pkl" ]; then
    echo "❌ Dataset not found: data/systematic_sweep/breakout_human_gameplay.pkl"
    echo "   Run the conversion script first:"
    echo "   python -m eb_jepa.datasets.atari.convert_human_to_systematic"
    exit 1
fi

echo "1. Dataset Structure Check"
echo "-------------------------------------------"
python verify_human_dataset.py

echo ""
echo ""
echo "2. Simulator Replay Verification (Trajectory 0)"
echo "-------------------------------------------"
python verify_human_replay.py --trajectory 0 --max_steps 500

echo ""
echo ""
echo "3. Simulator Replay Verification (Trajectory 1)"
echo "-------------------------------------------"
python verify_human_replay.py --trajectory 1 --max_steps 500

echo ""
echo "=========================================="
echo "✅ Verification complete!"
echo "=========================================="
