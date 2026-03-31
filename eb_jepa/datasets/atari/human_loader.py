"""
Human gameplay data loader for ATARI-HEAD dataset.

This module extracts and parses human gameplay data from the ATARI-HEAD dataset,
converting it into a format compatible with the systematic_sweep dataset structure.
"""

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import io


def extract_breakout_zip(zip_path: str = "data/human/breakout.zip",
                         output_dir: str = "data/human/") -> str:
    """
    Extract breakout.zip if not already extracted.

    Args:
        zip_path: Path to breakout.zip file
        output_dir: Directory to extract to

    Returns:
        Path to extracted breakout directory
    """
    extract_path = os.path.join(output_dir, "breakout")

    if os.path.exists(extract_path):
        print(f"✅ breakout/ already extracted at {extract_path}")
        return extract_path

    print(f"📦 Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"✅ Extracted to {extract_path}")
    return extract_path


def parse_label_file(label_path: str) -> pd.DataFrame:
    """
    Parse human gameplay label file.

    Label file format (comma-separated CSV):
        frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions

    Args:
        label_path: Path to .txt label file

    Returns:
        DataFrame with parsed labels
    """
    # Read only the first 6 columns (ignore variable-length gaze positions)
    # This avoids pandas parser errors from variable column counts
    df = pd.read_csv(
        label_path,
        usecols=[0, 1, 2, 3, 4, 5],
        names=['frame_id', 'episode_id', 'score', 'duration_ms', 'unclipped_reward', 'action'],
        skiprows=1  # Skip header row
    )

    # Add gaze_positions as empty string placeholder (we're not using it)
    df['gaze_positions'] = ''

    return df


def extract_frames_from_tar(tar_path: str, frame_ids: List[str]) -> np.ndarray:
    """
    Extract specific frames from .tar.bz2 archive.

    Args:
        tar_path: Path to .tar.bz2 file
        frame_ids: List of frame IDs (e.g., ['RZ_4137619_1', 'RZ_4137619_2', ...])

    Returns:
        Array of frames with shape [T, 210, 160, 3]
    """
    frames = []

    with tarfile.open(tar_path, 'r:bz2') as tar:
        # Get all members once to build a lookup dict
        members_dict = {}
        for member in tar.getmembers():
            if member.isfile():
                # Extract just the filename without path
                filename = Path(member.name).stem  # e.g., 'RZ_4137619_1'
                members_dict[filename] = member

        # Now extract frames in the order specified by frame_ids
        for frame_id in tqdm(frame_ids, desc=f"Extracting frames from {Path(tar_path).name}"):
            if frame_id in members_dict:
                member = members_dict[frame_id]
                f = tar.extractfile(member)

                # Load image
                img = Image.open(io.BytesIO(f.read()))
                frames.append(np.array(img))  # [210, 160, 3]
            else:
                print(f"Warning: Frame {frame_id} not found in tar archive, using black frame")
                # Use black frame as placeholder
                frames.append(np.zeros((210, 160, 3), dtype=np.uint8))

    return np.stack(frames)  # [T, 210, 160, 3]


# Breakout action mapping (ALE -> 0-3 discrete)
BREAKOUT_ACTION_MAP = {
    0: 0,   # NOOP
    1: 1,   # FIRE
    3: 2,   # RIGHT
    4: 3,   # LEFT (note: ALE uses 4 for LEFT)
}


def remap_actions(actions: np.ndarray) -> np.ndarray:
    """
    Remap ALE actions to 0-3 for Breakout.

    ALE action mapping:
        0 = NOOP
        1 = FIRE
        3 = RIGHT
        4 = LEFT

    Remapped to:
        0 = NOOP
        1 = FIRE
        2 = RIGHT
        3 = LEFT

    Args:
        actions: Array of ALE action integers

    Returns:
        Array of remapped actions in [0, 3]
    """
    remapped = np.zeros_like(actions, dtype=np.int64)
    for ale_action, breakout_action in BREAKOUT_ACTION_MAP.items():
        remapped[actions == ale_action] = breakout_action

    # Handle any unexpected actions
    if np.any((remapped < 0) | (remapped > 3)):
        print(f"Warning: Found unexpected actions. Unique values: {np.unique(actions)}")
        # Clip to valid range
        remapped = np.clip(remapped, 0, 3)

    return remapped


def extract_human_trial(trial_id: int,
                       base_path: str = "data/human/breakout",
                       metadata_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Extract frames, actions, rewards from a human trial.

    Args:
        trial_id: Trial ID number
        base_path: Path to extracted breakout data directory
        metadata_df: Optional metadata DataFrame (loaded from meta_data.csv)

    Returns:
        Dictionary containing:
            - frames: np.ndarray [T, 210, 160, 3] RGB frames
            - actions: np.ndarray [T] discrete action integers
            - rewards: np.ndarray [T] float rewards
            - trial_id: int
            - score: int (final score)
            - subject: str (subject ID)
            - episode_ids: np.ndarray [T] episode numbers
            - gaze_positions: List[str] (optional, for future use)
    """
    # Find files for this trial
    trial_files = list(Path(base_path).glob(f"{trial_id}_*.tar.bz2"))
    if not trial_files:
        raise FileNotFoundError(f"No .tar.bz2 file found for trial {trial_id} in {base_path}")

    tar_path = str(trial_files[0])

    # Corresponding label file
    label_path = tar_path.replace('.tar.bz2', '.txt')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    print(f"\nProcessing trial {trial_id}...")
    print(f"  TAR: {Path(tar_path).name}")
    print(f"  Labels: {Path(label_path).name}")

    # Parse labels
    labels_df = parse_label_file(label_path)

    # Extract data
    frame_ids = labels_df['frame_id'].values
    actions = labels_df['action'].values.astype(np.int64)
    rewards = labels_df['unclipped_reward'].values.astype(np.float32)
    episode_ids = labels_df['episode_id'].values.astype(np.int64) if 'episode_id' in labels_df.columns else np.zeros(len(labels_df), dtype=np.int64)
    scores = labels_df['score'].values.astype(np.int64) if 'score' in labels_df.columns else np.zeros(len(labels_df), dtype=np.int64)

    # Remap actions to 0-3
    actions = remap_actions(actions)

    # Extract frames from tar
    frames = extract_frames_from_tar(tar_path, frame_ids)

    # Get metadata
    if metadata_df is not None:
        trial_meta = metadata_df[metadata_df['trial_id'] == trial_id]
        if not trial_meta.empty:
            subject = trial_meta.iloc[0]['subject_id']
            final_score = trial_meta.iloc[0]['highest_score']
        else:
            subject = "Unknown"
            final_score = scores[-1] if len(scores) > 0 else 0
    else:
        subject = "Unknown"
        final_score = scores[-1] if len(scores) > 0 else 0

    print(f"  ✅ Extracted {len(frames)} frames")
    print(f"  Final score: {final_score}")
    print(f"  Episodes: {len(np.unique(episode_ids))}")

    return {
        'frames': frames,
        'actions': actions,
        'rewards': rewards,
        'trial_id': trial_id,
        'score': int(final_score),
        'subject': str(subject),
        'episode_ids': episode_ids,
        'gaze_positions': labels_df['gaze_positions'].tolist() if 'gaze_positions' in labels_df.columns else []
    }


def convert_to_trajectory_format(trial_data: Dict) -> List[Dict]:
    """
    Convert raw trial data into trajectory format matching systematic_sweep.

    Splits by episode if multiple episodes exist.

    Args:
        trial_data: Dictionary from extract_human_trial()

    Returns:
        List of trajectory dictionaries, each with:
            - observations: np.ndarray [T, 210, 160, 3]
            - actions: np.ndarray [T]
            - rewards: np.ndarray [T]
            - length: int
            - paddle_start_position: float (estimated from trial)
            - position_index: int (based on trial_id)
            - target_position: float (same as paddle_start_position)
            - paddle_policy: str ("human")
    """
    frames = trial_data['frames']
    actions = trial_data['actions']
    rewards = trial_data['rewards']
    episode_ids = trial_data['episode_ids']
    trial_id = trial_data['trial_id']

    # Split by episode
    unique_episodes = np.unique(episode_ids)
    trajectories = []

    for episode_idx in unique_episodes:
        # Get frames for this episode
        mask = episode_ids == episode_idx
        episode_frames = frames[mask]
        episode_actions = actions[mask]
        episode_rewards = rewards[mask]

        # Estimate paddle position (use trial_id as pseudo-position for now)
        # In real systematic sweep, paddle positions range from ~20-140 pixels
        # We'll distribute trials across this range
        paddle_position = 20 + (trial_id % 16) * 8  # Spread across 16 positions

        trajectory = {
            'observations': episode_frames,
            'actions': episode_actions,
            'rewards': episode_rewards,
            'length': len(episode_frames),
            'paddle_start_position': float(paddle_position),
            'position_index': int(trial_id % 100),  # Use trial_id as position index
            'target_position': float(paddle_position),
            'paddle_policy': 'human'
        }

        trajectories.append(trajectory)

        print(f"    Episode {episode_idx}: {len(episode_frames)} frames, "
              f"{episode_rewards.sum():.0f} total reward")

    return trajectories
