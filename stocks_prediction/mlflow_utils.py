"""MLflow helpers – logging config, git commit, and metric plots."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient


logger = logging.getLogger(__name__)


def log_git_commit() -> None:
    """Attach the current git commit hash to the active MLflow run."""
    try:
        commit_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        mlflow.set_tag("git_commit", commit_id)
        mlflow.log_param("git_commit", commit_id)
        logger.info("Git commit id: %s", commit_id)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not fetch git commit: %s", exc)


def fetch_and_plot_metrics(
    client: MlflowClient,
    run_id: str,
    metric_keys: Iterable[str],
    dst_dir: Path,
) -> None:
    """Download metric history from MLflow and store PNG plots in *dst_dir*."""

    dst_dir.mkdir(parents=True, exist_ok=True)

    for key in metric_keys:
        history = client.get_metric_history(run_id, key)
        if not history:
            logging.warning("No metric history for key '%s'", key)
            continue
        steps = [p.step for p in history]
        values = [p.value for p in history]

        plt.figure()
        plt.plot(steps, values)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.title(key)
        plt.tight_layout()

        fig_path = dst_dir / f"{key}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        # add to MLflow artefacts for convenience --------------------------------
        mlflow.log_artifact(fig_path, artifact_path="plots")
