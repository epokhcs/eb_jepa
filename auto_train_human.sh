#!/bin/bash
# Auto-launch training once conversion completes

echo "Waiting for conversion to complete..."

# Wait for conversion process to finish
while ps aux | grep -q "[c]onvert_human_to_systematic"; do
    echo "  Still converting... ($(date +%H:%M:%S))"
    sleep 30
done

echo ""
echo "✅ Conversion complete!"
echo ""

# Verify the dataset structure
echo "Verifying dataset structure..."
python verify_human_dataset.py

echo ""
echo "=========================================="
echo "REPLAY VERIFICATION IN SIMULATOR"
echo "=========================================="
echo "Testing trajectory 0 (500 steps)..."
python verify_human_replay.py --trajectory 0 --max_steps 500

echo ""
echo "Testing trajectory 1 (500 steps)..."
python verify_human_replay.py --trajectory 1 --max_steps 500

echo ""
echo "=========================================="
echo "VERIFICATION COMPLETE"
echo "=========================================="
echo ""
echo "Starting training for 1 epoch..."
echo ""

# Launch training
python examples/ac_video_jepa/main.py \
    --cfg examples/ac_video_jepa/cfgs/train_atari_human.yaml \
    --device auto

echo ""
echo "✅ Training complete!"
