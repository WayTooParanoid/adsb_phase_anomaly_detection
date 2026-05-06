from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .airport_context import context_anomaly_for_phase
from .config import load_config
from .features import build_feature_matrix, records_to_features
from .io import load_adsb_file
from .model_scoring import (
    best_phase_model_candidate,
    calibrated_model_severity,
    calibrated_percentile_from_distance,
    calibrated_reason,
    phase_model_candidates_by_index,
    select_phase_model_candidate,
)
from .model_store import load_models


SCORE_COLUMNS = [
    "dbscan_cluster",
    "anomaly_score",
    "dbscan_distance",
    "calibrated_anomaly_score",
    "phase_distance_percentile",
    "behavior_scored",
    "behavior_anomaly",
    "behavior_reason",
    "dbscan_feature_contributions",
    "model_severity",
    "phase_candidates",
    "model_best_phase",
    "model_best_score",
    "model_best_calibrated_score",
    "model_best_cluster",
    "model_best_severity",
    "model_scored_phase",
    "context_anomaly",
    "context_reason",
    "is_anomaly",
    "reason",
]


def _with_score_columns(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for col in SCORE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(dtype=object)
    return out


def _attach_model_candidates(features: pd.DataFrame, models: dict, config: dict) -> tuple[pd.DataFrame, dict[Any, list[dict[str, object]]]]:
    candidates_by_index: dict[Any, list[dict[str, object]]] = phase_model_candidates_by_index(features, models, config)
    best_by_index = {idx: best_phase_model_candidate(candidates) for idx, candidates in candidates_by_index.items()}
    out = features.copy()
    out["phase_candidates"] = out.index.map(lambda idx: candidates_by_index.get(idx, []))
    out["model_best_phase"] = out.index.map(lambda idx: (best_by_index.get(idx) or {}).get("phase"))
    out["model_best_score"] = out.index.map(lambda idx: (best_by_index.get(idx) or {}).get("score"))
    out["model_best_calibrated_score"] = out.index.map(lambda idx: (best_by_index.get(idx) or {}).get("calibrated_score"))
    out["model_best_cluster"] = out.index.map(lambda idx: (best_by_index.get(idx) or {}).get("cluster"))
    out["model_best_severity"] = out.index.map(lambda idx: (best_by_index.get(idx) or {}).get("severity", "NO_MODEL"))
    return out, candidates_by_index


def _assign_phase_from_models(features: pd.DataFrame, candidates_by_index: dict[Any, list[dict[str, object]]], config: dict) -> pd.DataFrame:
    assignments = {idx: select_phase_model_candidate(candidates, config) for idx, candidates in candidates_by_index.items()}
    out = features.copy()
    out["phase"] = out.index.map(lambda idx: str(assignments[idx]["phase"]))
    out["phase_confidence"] = out.index.map(lambda idx: assignments[idx]["confidence"])
    out["phase_reason"] = out.index.map(lambda idx: assignments[idx]["reason"])
    out["phase_rule_candidates"] = out.index.map(lambda idx: [])
    context_results = out.apply(lambda row: context_anomaly_for_phase(row.to_dict(), str(row["phase"]), config), axis=1)
    out["context_anomaly"] = context_results.map(lambda item: bool(item[0]))
    out["context_reason"] = context_results.map(lambda item: item[1])
    return out


def _score_group_without_model(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["dbscan_cluster"] = "UNKNOWN_MODEL"
    group["anomaly_score"] = pd.NA
    group["dbscan_distance"] = pd.NA
    group["calibrated_anomaly_score"] = pd.NA
    group["phase_distance_percentile"] = pd.NA
    group["behavior_scored"] = False
    group["behavior_anomaly"] = False
    no_model_reason = group["phase_reason"] if "phase_reason" in group.columns else pd.Series("no_model_for_phase", index=group.index)
    group["behavior_reason"] = no_model_reason.fillna("no_model_for_phase")
    group["dbscan_feature_contributions"] = None
    group["model_severity"] = "NO_MODEL"
    group["is_anomaly"] = group["context_anomaly"].astype(bool)
    group["reason"] = group["context_reason"].where(group["context_anomaly"].astype(bool), group["behavior_reason"])
    return group


def _score_group_with_model(group: pd.DataFrame, artifact: dict, config: dict, *, assign_phase_from_models: bool) -> pd.DataFrame:
    group = group.copy()
    phase = str(group["model_scored_phase"].iloc[0])
    artifact_columns = artifact.get("feature_columns")
    if not artifact_columns:
        raise ValueError(f"Model artifact for phase {phase!r} does not declare feature_columns")
    imputation_values = artifact.get("imputation_values")
    if not imputation_values:
        raise ValueError(f"Model artifact for phase {phase!r} does not declare imputation_values")

    model = artifact["model"]
    X = build_feature_matrix(group, phase, artifact.get("config", config), artifact_columns, imputation_values)
    if len(getattr(model, "components_", [])) == 0:
        group["dbscan_cluster"] = "NO_CORE_MODEL"
        group["anomaly_score"] = pd.NA
        group["dbscan_distance"] = pd.NA
        group["calibrated_anomaly_score"] = pd.NA
        group["phase_distance_percentile"] = pd.NA
        group["behavior_scored"] = False
        group["behavior_anomaly"] = True
        group["behavior_reason"] = "model_has_no_core_samples"
        group["dbscan_feature_contributions"] = None
        group["model_severity"] = "NO_CORE"
    else:
        labels = model.predict_from_core_samples(X)
        distances = model.nearest_core_distance(X)
        scores = distances / max(float(model.eps), 1e-9)
        percentiles = [
            calibrated_percentile_from_distance(float(distance), artifact.get("distance_calibration", {}))
            for distance in distances
        ]
        group["dbscan_cluster"] = labels
        group["anomaly_score"] = scores
        group["dbscan_distance"] = distances
        group["calibrated_anomaly_score"] = percentiles
        group["phase_distance_percentile"] = percentiles
        group["dbscan_feature_contributions"] = model.nearest_core_feature_contributions(X, artifact_columns, top_n=3)
        group["behavior_scored"] = True
        group["behavior_anomaly"] = labels == -1
        group["model_severity"] = [
            calibrated_model_severity(int(label), percentile, float(score))
            for label, percentile, score in zip(labels, percentiles, scores)
        ]
        fallback_reason = pd.Series("outside_phase_dbscan_core", index=group.index)
        phase_reason = fallback_reason if assign_phase_from_models else group.get("phase_reason", fallback_reason)
        calibrated_reasons = pd.Series(
            [calibrated_reason(phase, percentile, float(score)) for percentile, score in zip(percentiles, scores)],
            index=group.index,
        )
        group["behavior_reason"] = calibrated_reasons.where(labels == -1, phase_reason.where(labels == -1, None))

    group["is_anomaly"] = group["context_anomaly"].astype(bool) | group["behavior_anomaly"].astype(bool)
    group["reason"] = group.apply(lambda row: row["context_reason"] if row["context_anomaly"] else row["behavior_reason"], axis=1)
    return group


def score_features(
    features: pd.DataFrame,
    models: dict,
    config: dict,
    *,
    assign_phase_from_models: bool = False,
) -> pd.DataFrame:
    if features.empty:
        return _with_score_columns(features)
    if "phase" not in features.columns and not assign_phase_from_models:
        return _with_score_columns(features)

    features, candidates_by_index = _attach_model_candidates(features, models, config)
    if assign_phase_from_models:
        features = _assign_phase_from_models(features, candidates_by_index, config)

    features["model_scored_phase"] = features["phase"]
    rows = []
    for phase, group in features.groupby("model_scored_phase", sort=False):
        group = group.copy()
        if "context_anomaly" not in group.columns:
            group["context_anomaly"] = False
        if "context_reason" not in group.columns:
            group["context_reason"] = None
        artifact = models.get(str(phase))
        if artifact is None:
            group = _score_group_without_model(group)
        else:
            group = _score_group_with_model(group, artifact, config, assign_phase_from_models=assign_phase_from_models)
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score ADS-B segments against saved phase DBSCAN models")
    parser.add_argument("--input")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="outputs/scored_segments.csv")
    parser.add_argument("--input-radius-nm", type=float, default=150.0)
    parser.add_argument("--limit-files", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not args.input:
        raise SystemExit("--input is required")
    config = load_config(args.config)
    raw = load_adsb_file(args.input, config=config, radius_nm=args.input_radius_nm, limit_files=args.limit_files)
    features = records_to_features(raw, config)
    models = load_models(args.models_dir)
    scored = score_features(features, models, config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.drop(columns=["points"], errors="ignore").to_csv(output, index=False)


if __name__ == "__main__":
    main()
