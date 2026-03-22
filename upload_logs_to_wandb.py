#!/usr/bin/env python3
"""
Utility to upload local training logs to Weights & Biases.

Usage:
    python upload_logs_to_wandb.py logs/training/impala_cov8_std16_simt12_idm1
    python upload_logs_to_wandb.py --all  # Upload all experiments
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

import fire


def upload_experiment_logs(
    log_dir: str,
    project: str = "eb_jepa",
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
    force_new_run: bool = False,
) -> None:
    """
    Upload local metrics logs to W&B.

    Args:
        log_dir: Path to experiment log directory (e.g., logs/training/exp_name)
        project: W&B project name
        run_name: Optional run name (defaults to experiment name)
        tags: Optional tags for the run
        force_new_run: If True, create new W&B run instead of resuming
    """
    # Load .env for W&B configuration
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env configuration")
    except ImportError:
        print("⚠️  python-dotenv not installed, using existing environment")

    # Check W&B API key
    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if not wandb_api_key:
        print("❌ Error: WANDB_API_KEY not set in .env file")
        print("   Add your API key to .env and try again")
        sys.exit(1)

    # Configure SSL verification
    verify_ssl = os.environ.get("WANDB_VERIFY_SSL", "true").strip().lower()
    disable_ssl = verify_ssl in ("false", "0", "no")

    if disable_ssl:
        print("⚠️  SSL certificate verification disabled")
        os.environ["WANDB_DISABLE_SSL_VERIFY"] = "true"
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass

    # Import wandb
    try:
        import wandb
    except ImportError:
        print("❌ Error: wandb not installed. Run: pip install wandb")
        sys.exit(1)

    # Validate log directory
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"❌ Error: Log directory not found: {log_dir}")
        sys.exit(1)

    # Find metrics files
    json_file = log_path / "training_latest.json"
    config_file = log_path / "config.json"

    if not json_file.exists():
        print(f"❌ Error: No metrics file found: {json_file}")
        sys.exit(1)

    print(f"\n📁 Loading logs from: {log_dir}")
    print(f"   Metrics file: {json_file.name}")
    print(f"   Config file: {config_file.name if config_file.exists() else 'not found'}")

    # Load metrics
    with open(json_file) as f:
        metrics_history = json.load(f)

    print(f"   Found {len(metrics_history)} logged data points")

    # Load config if available
    config_dict = {}
    if config_file.exists():
        with open(config_file) as f:
            config_dict = json.load(f)
        print(f"   Loaded experiment config")

    # Prepare W&B run name and tags
    if run_name is None:
        run_name = log_path.name

    if tags is None:
        tags = ["retroactive_upload"]

    # Check for existing W&B run ID
    wandb_id_file = log_path / "wandb_run_id.txt"
    resume_id = None
    if wandb_id_file.exists() and not force_new_run:
        with open(wandb_id_file) as f:
            resume_id = f.read().strip()
        print(f"\n🔄 Found existing W&B run ID: {resume_id}")
        print(f"   Will resume/update existing run")

    # Initialize W&B
    print(f"\n🚀 Initializing W&B...")
    print(f"   Project: {project}")
    print(f"   Run name: {run_name}")
    print(f"   Tags: {tags}")

    wandb_config = {
        "project": project,
        "name": run_name,
        "config": config_dict,
        "tags": tags,
    }

    if resume_id and not force_new_run:
        wandb_config["id"] = resume_id
        wandb_config["resume"] = "allow"

    try:
        run = wandb.init(**wandb_config)
        print(f"✅ W&B run initialized: {run.url}")

        # Save run ID for future resumes
        if not resume_id:
            with open(wandb_id_file, "w") as f:
                f.write(run.id)
            print(f"💾 Saved W&B run ID to: {wandb_id_file.name}")

    except wandb.errors.CommError as e:
        if "401" in str(e) or "PERMISSION_ERROR" in str(e):
            print(f"❌ Authentication Error: {e}")
            print("\n🔑 Your W&B API key may be invalid, expired, or revoked.")
            print("\nTo fix:")
            print("1. Get a fresh API key from: https://wandb.ai/authorize")
            print("2. Update your .env file:")
            print("   WANDB_API_KEY=<new_key>")
            print("3. Run this script again")
            sys.exit(1)
        else:
            print(f"❌ W&B Communication Error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing W&B: {e}")
        sys.exit(1)

    # Upload metrics
    print(f"\n📊 Uploading {len(metrics_history)} data points...")

    # Separate step-level metrics from epoch summaries
    step_metrics = []
    epoch_summaries = []

    for entry in metrics_history:
        if entry.get("type") == "epoch_summary":
            epoch_summaries.append(entry)
        else:
            step_metrics.append(entry)

    # Upload step-level metrics
    print(f"   Uploading {len(step_metrics)} step metrics...")
    for i, entry in enumerate(step_metrics):
        # Extract step number
        step = entry.get("step", i)

        # Prepare metrics (exclude metadata fields)
        log_data = {
            k: v for k, v in entry.items()
            if k not in ["timestamp", "step", "type"]
        }

        # Upload to W&B
        wandb.log(log_data, step=step)

        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(step_metrics):
            print(f"   Progress: {i + 1}/{len(step_metrics)}", end="\r")

    print()  # New line after progress

    # Upload epoch summaries
    if epoch_summaries:
        print(f"   Uploading {len(epoch_summaries)} epoch summaries...")
        for entry in epoch_summaries:
            epoch = entry.get("epoch", 0)
            log_data = {
                k: v for k, v in entry.items()
                if k not in ["timestamp", "type"]
            }
            # Use epoch-based step for summaries
            wandb.log(log_data, step=epoch * 31)  # Approximate step

    # Finish
    run.finish()
    print(f"\n✅ Upload complete!")
    print(f"   View at: {run.url}")


def upload_all_experiments(
    logs_base_dir: str = "logs/training",
    project: str = "eb_jepa",
) -> None:
    """
    Upload all experiment logs to W&B.

    Args:
        logs_base_dir: Base directory containing experiment folders
        project: W&B project name
    """
    logs_path = Path(logs_base_dir)
    if not logs_path.exists():
        print(f"❌ Error: Logs directory not found: {logs_base_dir}")
        sys.exit(1)

    # Find all experiment directories
    exp_dirs = [
        d for d in logs_path.iterdir()
        if d.is_dir() and (d / "training_latest.json").exists()
    ]

    if not exp_dirs:
        print(f"❌ No experiment logs found in: {logs_base_dir}")
        sys.exit(1)

    print(f"Found {len(exp_dirs)} experiments to upload:\n")
    for i, exp_dir in enumerate(exp_dirs, 1):
        print(f"{i}. {exp_dir.name}")

    print(f"\n{'=' * 60}")

    # Upload each experiment
    for i, exp_dir in enumerate(exp_dirs, 1):
        print(f"\n[{i}/{len(exp_dirs)}] Uploading: {exp_dir.name}")
        print("=" * 60)
        try:
            upload_experiment_logs(
                log_dir=str(exp_dir),
                project=project,
            )
        except Exception as e:
            print(f"❌ Error uploading {exp_dir.name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"✅ Uploaded {len(exp_dirs)} experiments to W&B")


def main(
    log_dir: Optional[str] = None,
    all: bool = False,
    project: str = "eb_jepa",
    run_name: Optional[str] = None,
    tags: Optional[str] = None,
    force_new: bool = False,
) -> None:
    """
    Upload local training logs to Weights & Biases.

    Args:
        log_dir: Path to experiment log directory (e.g., logs/training/exp_name)
        all: Upload all experiments in logs/training/
        project: W&B project name (default: eb_jepa)
        run_name: Optional custom run name
        tags: Comma-separated tags (e.g., "tag1,tag2,tag3")
        force_new: Force create new W&B run instead of resuming

    Examples:
        # Upload specific experiment
        python upload_logs_to_wandb.py logs/training/impala_cov8_std16_simt12_idm1

        # Upload all experiments
        python upload_logs_to_wandb.py --all

        # Custom project and tags
        python upload_logs_to_wandb.py logs/training/exp_name --project my_project --tags "test,v1"
    """
    if all:
        upload_all_experiments(project=project)
    elif log_dir:
        tag_list = tags.split(",") if tags else None
        upload_experiment_logs(
            log_dir=log_dir,
            project=project,
            run_name=run_name,
            tags=tag_list,
            force_new_run=force_new,
        )
    else:
        print("❌ Error: Specify either a log_dir or use --all")
        print("\nUsage:")
        print("  python upload_logs_to_wandb.py logs/training/exp_name")
        print("  python upload_logs_to_wandb.py --all")
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
