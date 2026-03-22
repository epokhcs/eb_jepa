# Human Performance Dataset for ATARI Breakout

## Overview

This directory contains high-quality human gameplay data for ATARI games, including **Breakout**, collected with simultaneous eye-tracking. This dataset serves as a gold standard for:

1. **Validation**: Compare JEPA model predictions against real human gameplay
2. **Benchmarking**: Measure how close the model's world understanding is to human performance
3. **Planning**: Use human action sequences as expert demonstrations
4. **Analysis**: Understand what humans attend to (via gaze data) vs what the model focuses on

## Dataset Statistics

- **Total size**: 117 hours of gameplay across 20 ATARI games
- **Action demonstrations**: 8 million human actions
- **Gaze samples**: 328 million eye-tracking samples
- **Quality**: Near-optimal decisions with scores comparable to human records
- **Collection method**: Frame-by-frame gameplay (resolves state-action mismatch)

### Breakout Specific Data

From `meta_data.csv`, the Breakout dataset includes:
- Multiple human subjects playing Breakout
- Scores ranging from ~300-450 (competitive human performance)
- Complete action sequences with reaction times
- Eye-tracking gaze positions for each frame

## Files in this Directory

```
data/human/
├── README.md           # Original dataset description
├── USAGE.md           # This file - usage guide for eb_jepa
├── meta_data.csv      # Metadata for all trials (471 trials total)
├── action_enums.txt   # ALE action integer to name mapping
└── breakout.zip       # Breakout game data (126 MB) - DOWNLOAD REQUIRED
    ├── *.tar.bz2      # Game frames (images)
    └── *.txt          # Labels (actions, rewards, gaze)
```

## Download Dataset

⚠️ **The `breakout.zip` file (126 MB) is not included in this repository due to size constraints.**

**Download from original source:**
```bash
# Download ATARI-HEAD Breakout dataset
cd data/human
wget https://zenodo.org/record/3451402/files/breakout.zip
# Or manually download from: https://zenodo.org/record/3451402
```

Alternatively, the full ATARI-HEAD dataset is available at:
- **Zenodo**: https://zenodo.org/record/3451402
- **Paper**: https://arxiv.org/abs/1903.06754

## Data Format

### meta_data.csv columns:
- `GameName`: Game identifier (e.g., "breakout")
- `trial_id`: Unique trial identifier
- `subject_id`: Human subject ID
- `highest_score`: Maximum score achieved
- `total_frame`: Number of frames in trial
- `total_game_play_time`: Duration in milliseconds
- `total_episode`: Number of episodes (lives consumed)
- `avg_error`, `max_error`: Eye-tracking calibration errors
- `low_sample_rate`: % frames with insufficient gaze samples
- `fps`: Frames per second

### Label files (*.txt in breakout.zip):
Each frame has:
- `frame_id`: Frame identifier (links to image in .tar.bz2)
- `episode_id`: Episode number within trial
- `score`: Current game score
- `duration(ms)`: Human reaction time for this action
- `unclipped_reward`: Immediate reward from environment
- `action`: Action integer (see action_enums.txt)
- `gaze_positions`: Eye-tracking coordinates `x0,y0,x1,y1,...`

### Action Mapping (action_enums.txt):

For Breakout, the relevant actions are:
```
0  = NOOP    (no action)
1  = FIRE    (launch ball)
3  = RIGHT   (move paddle right)
4  = LEFT    (move paddle left)
```

## Usage with JEPA Model

### 1. Extract Breakout Data

```bash
cd data/human
unzip breakout.zip
```

This creates:
- `breakout/[trial_id].tar.bz2` - Image frames
- `breakout/[trial_id].txt` - Action and gaze labels

### 2. Compare Model Predictions to Human Performance

```python
import pandas as pd

# Load Breakout trials
metadata = pd.read_csv('data/human/meta_data.csv')
breakout_trials = metadata[metadata['GameName'] == 'breakout']

print(f"Number of Breakout trials: {len(breakout_trials)}")
print(f"Average score: {breakout_trials['highest_score'].mean():.1f}")
print(f"Best score: {breakout_trials['highest_score'].max()}")
print(f"Total frames: {breakout_trials['total_frame'].sum():,}")
```

### 3. Load Human Trajectories for Validation

```python
import tarfile
import bz2

def load_human_trajectory(trial_id):
    """Load frames and actions from a human trial."""
    # Extract frames
    with tarfile.open(f'data/human/breakout/{trial_id}.tar.bz2', 'r:bz2') as tar:
        frames = []
        for member in sorted(tar.getmembers()):
            f = tar.extractfile(member)
            frame = np.frombuffer(f.read(), dtype=np.uint8)
            frames.append(frame.reshape(210, 160, 3))

    # Load labels
    labels = pd.read_csv(f'data/human/breakout/{trial_id}.txt')
    actions = labels['action'].values
    rewards = labels['unclipped_reward'].values

    return frames, actions, rewards

# Example: Load first Breakout trial
trial_id = breakout_trials.iloc[0]['trial_id']
frames, actions, rewards = load_human_trajectory(trial_id)
print(f"Loaded {len(frames)} frames with {len(actions)} actions")
```

### 4. Validation Metrics

Compare JEPA model predictions against human data:

```python
def evaluate_model_vs_human(jepa_model, human_trajectory):
    """
    Evaluate how well JEPA predictions match human gameplay.

    Metrics:
    1. Frame prediction accuracy (MSE between predicted and actual next frames)
    2. Action prediction accuracy (compare IDM predictions to human actions)
    3. Reward prediction accuracy (compare predicted vs actual rewards)
    4. Trajectory divergence (how quickly predicted trajectory deviates)
    """
    human_frames, human_actions, human_rewards = human_trajectory

    # Predict future frames given initial state + human actions
    predicted_frames = jepa_model.unroll(
        obs_init=human_frames[0],
        actions=human_actions,
        nsteps=len(human_actions)
    )

    # Compute metrics
    frame_mse = ((predicted_frames - human_frames[1:]) ** 2).mean()

    # Inverse dynamics: predict actions from state transitions
    predicted_actions = jepa_model.idm(
        human_frames[:-1],
        human_frames[1:]
    )
    action_accuracy = (predicted_actions == human_actions).mean()

    return {
        'frame_mse': frame_mse,
        'action_accuracy': action_accuracy,
        'avg_human_reward': human_rewards.mean()
    }
```

### 5. Benchmark Planning Algorithms

Use human actions as optimal baseline:

```python
def benchmark_planner(jepa_model, planner, human_trajectory):
    """
    Compare planner's actions to human expert demonstrations.

    Returns:
    - Score achieved by planner vs human score
    - Action agreement rate with human
    - Trajectory efficiency
    """
    human_frames, human_actions, _ = human_trajectory

    # Run planner from same initial state
    planned_actions = planner.plan(
        initial_state=human_frames[0],
        horizon=len(human_actions)
    )

    # Compare
    action_agreement = (planned_actions == human_actions).mean()

    # Execute planner's actions in env
    planner_score = execute_in_env(planned_actions)
    human_score = human_rewards.sum()

    return {
        'planner_score': planner_score,
        'human_score': human_score,
        'action_agreement': action_agreement
    }
```

## Key Insights from Human Data

### Why Frame-by-Frame Gameplay?

The dataset uses **frame-by-frame** collection (not real-time) to:

1. **Resolve state-action mismatch**: Human reaction time is 250-300ms, so real-time recording causes misalignment between state and action. Frame-by-frame ensures perfect alignment.

2. **Maximize performance**: Reduces fatigue and inattentive blindness, leading to near-optimal decisions.

3. **Capture all eye movements**: Critical states requiring planning show multiple gaze fixations. This data reveals what humans attend to when making decisions.

### Human Performance Baselines

From the metadata:
- **Breakout scores**: 300-450 (expert human performance)
- **Reaction times**: Available via `duration(ms)` field
- **Gaze patterns**: Shows what parts of screen humans focus on

## Citation

If you use this dataset, please cite:

```
Zhang, R., et al. "Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset"
AAAI Conference on Artificial Intelligence, 2020.
```

## Applications in eb_jepa

### Current Use Cases:
1. ✅ **Validation dataset**: Compare JEPA predictions to human gameplay
2. ✅ **Benchmarking**: Measure prediction quality against ground truth
3. ✅ **Action analysis**: Evaluate IDM predictions vs human actions

### Future Use Cases:
- **Imitation learning**: Train policy to mimic human actions
- **Attention mechanisms**: Learn from human gaze to improve state representations
- **Planning evaluation**: Use human trajectories as optimal baseline
- **Curriculum learning**: Progressive training from easy to hard human demonstrations

## Example: Quick Validation Script

```bash
# Extract Breakout data
cd data/human && unzip -q breakout.zip && cd ../..

# Run validation
python examples/ac_video_jepa/validate_human.py \
  --checkpoint checkpoints/latest.pth.tar \
  --config checkpoints/config.yaml \
  --human_data data/human \
  --game breakout \
  --num_trials 10
```

This compares your trained JEPA model predictions against 10 human Breakout trials.

## Notes

- **Data size**: The full `breakout.zip` is 126 MB (compressed)
- **Frame format**: RGB images, 210x160 resolution
- **Eye-tracking**: Coordinates in pixel space (0,0) = top-left
- **Action space**: Consistent with ALE (Arcade Learning Environment)

## See Also

- [Original README](README.md) - Full dataset description
- [Dataset Paper](https://arxiv.org/abs/1903.06754) - ATARI-HEAD paper
- [eb_jepa Documentation](../../README.md) - Main project README
