#!/usr/bin/env python3
"""
Analyze paddle positions from systematic sweep heatmap.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the heatmap data
data = np.load('data/systematic_sweep/breakout_heatmaps.npz')
paddle_heatmap = data['paddle_heatmap']
ball_heatmap = data['ball_heatmap']

print("=" * 70)
print("PADDLE POSITION ANALYSIS")
print("=" * 70)
print(f"Heatmap shape: {paddle_heatmap.shape}")
print(f"Total paddle visits: {paddle_heatmap.sum():,}")
print()

# Analyze paddle x-position distribution
# Sum over y-axis to get visits per x-position
paddle_x_distribution = paddle_heatmap.sum(axis=0)

# Find positions with significant visits (threshold: > 100 visits)
threshold = 100
significant_positions = np.where(paddle_x_distribution > threshold)[0]

print(f"X-positions with > {threshold} visits:")
print(f"Total x-positions: {len(significant_positions)}")
print()

# Group nearby positions (within 2 pixels) to find distinct paddle positions
paddle_positions = []
current_pos_start = None
current_pos_visits = 0

for x in significant_positions:
    if current_pos_start is None:
        current_pos_start = x
        current_pos_visits = paddle_x_distribution[x]
    elif x - current_pos_start <= 2:  # Within paddle width
        current_pos_visits += paddle_x_distribution[x]
    else:
        # New paddle position
        center_x = current_pos_start + 1  # Approximate center
        paddle_positions.append((center_x, current_pos_visits))
        current_pos_start = x
        current_pos_visits = paddle_x_distribution[x]

# Add last position
if current_pos_start is not None:
    center_x = current_pos_start + 1
    paddle_positions.append((center_x, current_pos_visits))

print(f"Detected distinct paddle positions: {len(paddle_positions)}")
print()
print("Position | X-coord | Total Visits | % of Total")
print("-" * 60)

total_visits = sum(visits for _, visits in paddle_positions)
for i, (x, visits) in enumerate(paddle_positions):
    pct = 100 * visits / total_visits
    print(f"{i:4d}     | {x:7d} | {visits:12,} | {pct:6.2f}%")

# Calculate spacing between positions
if len(paddle_positions) > 1:
    print()
    print("Spacing between consecutive positions:")
    print("-" * 40)
    for i in range(len(paddle_positions) - 1):
        x1 = paddle_positions[i][0]
        x2 = paddle_positions[i+1][0]
        spacing = x2 - x1
        print(f"Position {i} → {i+1}: {spacing} pixels")

    spacings = [paddle_positions[i+1][0] - paddle_positions[i][0]
                for i in range(len(paddle_positions) - 1)]
    print()
    print(f"Average spacing: {np.mean(spacings):.1f} pixels")
    print(f"Min spacing: {min(spacings)} pixels")
    print(f"Max spacing: {max(spacings)} pixels")
    print(f"Std dev: {np.std(spacings):.2f} pixels")

# Visualize
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: X-position distribution
ax1.bar(range(len(paddle_x_distribution)), paddle_x_distribution, width=1, color='orange')
ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
ax1.set_title('Paddle X-Position Distribution (Full)', fontsize=14, fontweight='bold')
ax1.set_xlabel('X Position (pixels)', fontsize=12)
ax1.set_ylabel('Total Visits', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Detected paddle positions
positions_x = [x for x, _ in paddle_positions]
positions_visits = [visits for _, visits in paddle_positions]
ax2.bar(range(len(paddle_positions)), positions_visits, color='green', alpha=0.7)
ax2.set_title(f'Detected {len(paddle_positions)} Distinct Paddle Positions', fontsize=14, fontweight='bold')
ax2.set_xlabel('Position Index', fontsize=12)
ax2.set_ylabel('Total Visits', fontsize=12)
ax2.set_xticks(range(len(paddle_positions)))
ax2.set_xticklabels([f"{i}\n(x={x})" for i, (x, _) in enumerate(paddle_positions)],
                      rotation=0, fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = Path('visualizations/paddle_position_analysis.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print()
print(f"✅ Visualization saved to: {output_path}")
print("=" * 70)
