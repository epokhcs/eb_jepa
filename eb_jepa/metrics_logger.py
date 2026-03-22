"""
Metrics logging utilities for training and inference performance tracking.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from eb_jepa.logging import get_logger

logger = get_logger(__name__)


class MetricsLogger:
    """Logger for training and inference metrics with JSON and CSV export."""

    def __init__(
        self,
        log_dir: Path,
        experiment_name: str,
        mode: str = "training",
    ):
        """
        Initialize metrics logger.

        Args:
            log_dir: Base directory for logs (e.g., logs/training or logs/inference)
            experiment_name: Name of the experiment
            mode: Either "training" or "inference"
        """
        self.mode = mode
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize metric storage
        self.metrics_history = []

        # File paths
        self.json_path = self.log_dir / f"{mode}_{self.run_timestamp}.json"
        self.csv_path = self.log_dir / f"{mode}_{self.run_timestamp}.csv"
        self.latest_json = self.log_dir / f"{mode}_latest.json"
        self.latest_csv = self.log_dir / f"{mode}_latest.csv"

        logger.info(f"📊 Metrics logging to: {self.log_dir}")

    def log_metrics(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> None:
        """
        Log metrics for a training/inference step.

        Args:
            epoch: Current epoch number
            step: Current global step
            metrics: Dictionary of metric name -> value
            prefix: Optional prefix for metric names
        """
        # Add timestamp and metadata
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "step": step,
        }

        # Add metrics with optional prefix
        for key, value in metrics.items():
            metric_name = f"{prefix}{key}" if prefix else key
            entry[metric_name] = float(value) if isinstance(value, (int, float)) else value

        # Store in memory
        self.metrics_history.append(entry)

        # Save incrementally (append to files)
        self._save_metrics()

    def log_epoch_summary(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        elapsed_time: Optional[float] = None,
    ) -> None:
        """
        Log end-of-epoch summary metrics.

        Args:
            epoch: Epoch number
            metrics: Summary metrics (e.g., avg_loss, val_loss, etc.)
            elapsed_time: Epoch duration in seconds
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "type": "epoch_summary",
        }

        if elapsed_time is not None:
            entry["epoch_time_seconds"] = elapsed_time

        entry.update({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})

        self.metrics_history.append(entry)
        self._save_metrics()

    def _save_metrics(self) -> None:
        """Save metrics to JSON and CSV files."""
        if not self.metrics_history:
            return

        # Save as JSON
        with open(self.json_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        # Create symlink to latest
        if self.latest_json.exists() or self.latest_json.is_symlink():
            self.latest_json.unlink()
        self.latest_json.symlink_to(self.json_path.name)

        # Save as CSV
        try:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(self.csv_path, index=False)

            # Create symlink to latest
            if self.latest_csv.exists() or self.latest_csv.is_symlink():
                self.latest_csv.unlink()
            self.latest_csv.symlink_to(self.csv_path.name)
        except Exception as e:
            logger.warning(f"Failed to save CSV: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged metrics."""
        if not self.metrics_history:
            return {}

        df = pd.DataFrame(self.metrics_history)

        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=["number"]).columns

        summary = {}
        for col in numeric_cols:
            if col not in ["epoch", "step"]:
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
                summary[f"{col}_min"] = df[col].min()
                summary[f"{col}_max"] = df[col].max()

        return summary

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration."""
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"💾 Saved config to: {config_path}")

    def close(self) -> None:
        """Finalize logging and save summary."""
        self._save_metrics()

        summary = self.get_summary()
        if summary:
            summary_path = self.log_dir / f"{self.mode}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"📈 Saved summary to: {summary_path}")


def setup_wandb_from_env(
    project: str,
    config: Dict[str, Any],
    run_dir: Path,
    run_name: Optional[str] = None,
    tags: Optional[list] = None,
    group: Optional[str] = None,
) -> Optional[Any]:
    """
    Setup wandb with automatic API key detection from .env file.

    Args:
        project: W&B project name
        config: Experiment configuration
        run_dir: Run directory
        run_name: Optional run name
        tags: Optional tags
        group: Optional group name

    Returns:
        wandb run object if enabled, None otherwise
    """
    # Try to load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.info("python-dotenv not installed, skipping .env loading")

    # Check if WANDB_API_KEY is set
    wandb_api_key = os.environ.get("WANDB_API_KEY", "").strip()
    wandb_mode = os.environ.get("WANDB_MODE", "").strip().lower()

    # Check SSL verification setting (defaults to true for security)
    verify_ssl = os.environ.get("WANDB_VERIFY_SSL", "true").strip().lower()
    disable_ssl = verify_ssl in ("false", "0", "no")

    # Configure SSL verification for W&B
    if disable_ssl:
        logger.info("⚠️  SSL certificate verification disabled for W&B")
        os.environ["WANDB_DISABLE_SSL_VERIFY"] = "true"
        # Also disable SSL warnings to reduce noise
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass
    else:
        # Ensure SSL verification is enabled (default behavior)
        if "WANDB_DISABLE_SSL_VERIFY" in os.environ:
            del os.environ["WANDB_DISABLE_SSL_VERIFY"]

    # If API key is present and mode is not disabled, enable wandb
    if wandb_api_key and wandb_mode not in ("disabled", "offline"):
        logger.info("🔑 WANDB_API_KEY detected, enabling wandb logging")
        enabled = True
    else:
        if wandb_mode in ("disabled", "offline"):
            logger.info("📴 W&B disabled via WANDB_MODE")
        else:
            logger.info("📴 W&B disabled (no API key in .env)")
        enabled = False

    # Use existing setup_wandb function
    from eb_jepa.training_utils import setup_wandb

    return setup_wandb(
        project=project,
        config=config,
        run_dir=run_dir,
        run_name=run_name,
        tags=tags,
        group=group,
        enabled=enabled,
    )
