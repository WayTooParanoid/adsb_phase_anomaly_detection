from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import joblib
import numpy as np

from .config import phase_tolerances
from .features import feature_columns
from .phase_classifier import PHASES


def training_summary(labels, n_core_samples: int) -> dict[str, Any]:
    labels = np.asarray(labels)
    cluster_labels = sorted(int(x) for x in set(labels.tolist()) if int(x) >= 0)
    return {
        "n_segments": int(len(labels)),
        "n_clusters": int(len(cluster_labels)),
        "noise_rate": float(np.mean(labels == -1)) if len(labels) else 1.0,
        "n_core_samples": int(n_core_samples),
    }


def save_model_artifact(
    output_dir: str | Path,
    phase: str,
    config: dict,
    model,
    summary: dict[str, Any],
    feature_columns_override: list[str] | None = None,
    imputation_values: dict[str, float] | None = None,
    dbscan_params: dict[str, Any] | None = None,
    distance_calibration: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "phase": phase,
        "config": config,
        "feature_columns": feature_columns_override or feature_columns(),
        "imputation_values": imputation_values or {},
        "feature_family": "aviation_behavior",
        "training_config_version": 2,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "training_summary": summary,
        "dbscan_params": dbscan_params or {},
        "distance_calibration": distance_calibration or {},
        "tolerances": phase_tolerances(config, phase),
    }
    path = output_dir / f"{phase.lower()}_dbscan.joblib"
    joblib.dump(artifact, path)
    return path


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def load_models(models_dir: str | Path) -> dict[str, dict[str, Any]]:
    models = {}
    active_phases = set(PHASES)
    for path in Path(models_dir).glob("*_dbscan.joblib"):
        artifact = load_model_artifact(path)
        phase = artifact["phase"]
        if phase in active_phases:
            models[phase] = artifact
    return models
