from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, phase_tolerances
from .custom_dbscan import CustomDBSCAN
from .features import (
    build_feature_matrix,
    compute_imputation_values,
    feature_columns,
    records_to_features,
    rows_with_required_features,
)
from .io import load_adsb_file, save_json
from .phase_classifier import classify_segments
from .model_store import save_model_artifact, training_summary

LOGGER = logging.getLogger(__name__)


def _fit_model(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    *,
    min_unique_tracks: int | None = None,
    group_ids=None,
    feature_weights: np.ndarray | None = None,
) -> CustomDBSCAN:
    return CustomDBSCAN(
        eps=eps,
        min_samples=min_samples,
        min_unique_tracks=min_unique_tracks,
        feature_weights=feature_weights,
    ).fit(X, group_ids=group_ids)


def _group_ids_for_training(df: pd.DataFrame) -> pd.Series:
    for column in ("flight_id", "track_id", "hex"):
        if column in df.columns:
            values = df[column].astype(str).replace("", pd.NA)
            if values.notna().all():
                return values
    raise ValueError("Track-aware DBSCAN requires flight_id, track_id, or hex values")


def _min_unique_tracks_for_phase(config: dict, phase: str) -> int | None:
    value = config["dbscan"].get("min_unique_tracks_by_phase", {}).get(phase)
    if value is None:
        return None
    value = int(value)
    if value <= 1:
        return None
    return value


def _feature_weights_for_phase(config: dict, phase: str, columns: list[str]) -> np.ndarray | None:
    raw = config["dbscan"].get("feature_weights_by_phase", {}).get(phase)
    if not raw:
        return None
    unknown = sorted(set(raw) - set(columns))
    if unknown:
        raise ValueError(f"Unknown DBSCAN feature weight(s) for phase {phase}: {unknown}")
    weights = np.array([float(raw.get(column, 1.0)) for column in columns], dtype=float)
    CustomDBSCAN._validate_feature_weights(weights, len(columns))
    return weights


def _training_scope_mask(features: pd.DataFrame, config: dict) -> pd.Series:
    mask = pd.Series(True, index=features.index)
    dbscan_cfg = config.get("dbscan", {})
    training_phases = dbscan_cfg.get("training_phases")
    if training_phases and "phase" in features.columns:
        mask &= features["phase"].isin(set(map(str, training_phases)))
    training_filter = dbscan_cfg.get("training_filter", {})
    if training_filter.get("primary_airport_only") and "airport_id" in features.columns:
        primary_id = str(config.get("airport", {}).get("id") or "").upper()
        mask &= features["airport_id"].fillna("").astype(str).str.upper().eq(primary_id)
    context_types = training_filter.get("airport_context_types")
    if context_types and "airport_context_type" in features.columns:
        mask &= features["airport_context_type"].fillna("").astype(str).isin(set(map(str, context_types)))
    return mask


def _weighted_distance(diff: np.ndarray, metric: str, feature_weights: np.ndarray | None = None) -> np.ndarray:
    if metric == "weighted_manhattan":
        weights = 1.0 if feature_weights is None else feature_weights
        return np.sum(weights * np.abs(diff), axis=-1)
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=-1)
    if metric == "weighted_euclidean":
        weights = 1.0 if feature_weights is None else feature_weights
        return np.sqrt(np.sum(weights * diff * diff, axis=-1))
    if metric == "chebyshev":
        return np.max(np.abs(diff), axis=-1)
    if metric == "euclidean":
        return np.linalg.norm(diff, axis=-1)
    raise ValueError(f"Unsupported DBSCAN distance metric: {metric}")


def _weighted_kth_neighbor_distances(
    X: np.ndarray,
    k: int,
    feature_weights: np.ndarray | None = None,
    batch_size: int = 512,
    metric: str = "weighted_euclidean",
) -> np.ndarray:
    if len(X) < k:
        raise ValueError("k-distance eps selection requires at least k samples")
    kth = np.empty(len(X), dtype=float)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        diff = X[start:stop, None, :] - X[None, :, :]
        distances = _weighted_distance(diff, metric, feature_weights)
        kth[start:stop] = np.partition(distances, k - 1, axis=1)[:, k - 1]
    return kth


def select_eps_for_phase(X: np.ndarray, min_samples: int, config: dict, feature_weights: np.ndarray | None = None) -> float:
    selection = config["dbscan"].get("eps_selection", {})
    if not selection.get("enabled", False):
        return float(config["dbscan"]["default_eps"])
    method = selection.get("method", "kth_neighbor_percentile")
    if method != "kth_neighbor_percentile":
        raise ValueError(f"Unsupported eps_selection method: {method}")
    percentile = float(selection.get("percentile", 90))
    if not 0 < percentile <= 100:
        raise ValueError("eps_selection percentile must be in (0, 100]")
    distances = _weighted_kth_neighbor_distances(X, int(min_samples), feature_weights)
    eps = float(np.percentile(distances, percentile))
    min_eps = selection.get("min_eps")
    max_eps = selection.get("max_eps")
    if min_eps is not None:
        eps = max(eps, float(min_eps))
    if max_eps is not None:
        eps = min(eps, float(max_eps))
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps_selection produced a non-positive or non-finite eps")
    return eps


def _sample_for_dbscan(df: pd.DataFrame, max_rows: int | None, seed: int = 42) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).sort_values("start_time").reset_index(drop=True)


def clear_model_artifacts(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_artifact in output_dir.glob("*_dbscan.joblib"):
        old_artifact.unlink()


def _distance_calibration(model: CustomDBSCAN, X: np.ndarray, labels: np.ndarray) -> dict:
    distances = model.nearest_core_distance(X)
    finite = distances[np.isfinite(distances)]
    inlier_distances = distances[(np.asarray(labels) != -1) & np.isfinite(distances)]
    reference = inlier_distances if len(inlier_distances) else finite
    if len(reference) == 0:
        return {
            "reference": "none",
            "n_reference": 0,
            "n_noise_training_segments": int(np.sum(np.asarray(labels) == -1)),
            "percentiles": {},
            "reference_distances": [],
        }

    reference = np.sort(reference.astype(float))
    percentiles = {
        "p50": float(np.percentile(reference, 50)),
        "p75": float(np.percentile(reference, 75)),
        "p90": float(np.percentile(reference, 90)),
        "p95": float(np.percentile(reference, 95)),
        "p97_5": float(np.percentile(reference, 97.5)),
        "p99": float(np.percentile(reference, 99)),
    }
    return {
        "reference": "dbscan_inliers" if len(inlier_distances) else "all_finite_training_distances",
        "n_reference": int(len(reference)),
        "n_noise_training_segments": int(np.sum(np.asarray(labels) == -1)),
        "percentiles": percentiles,
        "reference_distances": reference.tolist(),
    }


def _compact_distance_calibration(calibration: dict) -> dict:
    return {key: value for key, value in calibration.items() if key != "reference_distances"}


def _read_feature_cache(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _looks_like_feature_cache(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz") or path.suffix.lower() in {".parquet", ".pq"}):
        return False
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            columns = set(pd.read_parquet(path).columns)
            required = {"phase", *feature_columns()}
            return required.issubset(columns)
        sample = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    required = {"phase", *feature_columns()}
    return required.issubset(sample.columns)


def train_phase_models(features: pd.DataFrame, config: dict, output_dir: str | Path, max_segments_per_phase: int | None = None) -> list[dict]:
    clear_model_artifacts(output_dir)
    if features.empty or "phase" not in features.columns:
        raise ValueError("No phase-labeled segments available for DBSCAN training")
    total_segments = len(features)
    trainable = features[features["phase"].notna()].copy()
    if not config["dbscan"].get("train_unknown_phase"):
        trainable = trainable[trainable["phase"] != "UNKNOWN"]
    trainable = trainable[_training_scope_mask(trainable, config)]
    if "context_anomaly" in trainable.columns:
        trainable = trainable[~trainable["context_anomaly"].astype(bool)]
    trainable = trainable[rows_with_required_features(trainable)]
    LOGGER.info("DBSCAN clean training rows: %s/%s", len(trainable), total_segments)
    if trainable.empty:
        raise ValueError("No context-clean segments with required aviation features are available for DBSCAN training")
    summaries = []
    phases = sorted(trainable["phase"].dropna().unique())
    for phase in phases:
        all_phase_df = trainable[trainable["phase"] == phase].reset_index(drop=True)
        phase_df = _sample_for_dbscan(all_phase_df, max_segments_per_phase)
        min_samples = int(config["dbscan"]["min_samples_by_phase"].get(phase, 10))
        if len(phase_df) < min_samples:
            LOGGER.warning("Skipping %s: %s segments < min_samples %s", phase, len(phase_df), min_samples)
            continue
        columns = feature_columns()
        imputation_values = compute_imputation_values(phase_df, columns)
        X = build_feature_matrix(phase_df, phase, config, columns, imputation_values)
        min_unique_tracks = _min_unique_tracks_for_phase(config, phase)
        group_ids = _group_ids_for_training(phase_df) if min_unique_tracks else None
        feature_weights = _feature_weights_for_phase(config, phase, columns)
        eps = select_eps_for_phase(X, min_samples, config, feature_weights)
        LOGGER.info("Training %s DBSCAN with eps=%s min_samples=%s min_unique_tracks=%s", phase, f"{eps:.4f}", min_samples, min_unique_tracks or 1)
        model = _fit_model(
            X,
            eps,
            min_samples,
            min_unique_tracks=min_unique_tracks,
            group_ids=group_ids,
            feature_weights=feature_weights,
        )
        summary = training_summary(model.labels_, len(model.core_sample_indices_))
        if summary["n_core_samples"] == 0:
            LOGGER.warning("Skipping %s: fitted DBSCAN has no core samples", phase)
            continue
        calibration = _distance_calibration(model, X, model.labels_)
        summary["selected_eps"] = float(eps)
        summary["min_unique_tracks"] = int(min_unique_tracks or 1)
        summary["n_available_segments"] = int(len(features[features["phase"] == phase]))
        summary["n_clean_training_segments"] = int(len(all_phase_df))
        summary["distance_calibration"] = _compact_distance_calibration(calibration)
        save_model_artifact(
            output_dir,
            phase,
            config,
            model,
            summary,
            columns,
            imputation_values,
            dbscan_params={
                "eps": float(eps),
                "min_samples": int(min_samples),
                "min_unique_tracks": int(min_unique_tracks or 1),
                "feature_weights": feature_weights.tolist() if feature_weights is not None else None,
            },
            distance_calibration=calibration,
        )
        summaries.append({"phase": phase, **summary})
    return summaries


def run_training(args: argparse.Namespace) -> pd.DataFrame:
    config = load_config(args.config)
    if _looks_like_feature_cache(args.input):
        features = _read_feature_cache(args.input)
        training_source = {"kind": "feature_cache", "path": str(Path(args.input).resolve())}
        LOGGER.info("Loaded training feature cache %s with %s segments", args.input, len(features))
        features = classify_segments(features, config)
        LOGGER.info("Refreshed phase labels from current classifier rules")
    else:
        raw = load_adsb_file(args.input, config=config, radius_nm=args.input_radius_nm, limit_files=args.limit_files)
        features = records_to_features(raw, config)
        training_source = {"kind": "raw_adsb", "path": str(Path(args.input).resolve())}
    if features.empty:
        raise RuntimeError("No valid trajectory segments were produced")

    output_dir = Path(args.output_dir)
    outputs_dir = output_dir.parent / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    summaries = train_phase_models(features, config, output_dir, args.max_segments_per_phase)

    features.drop(columns=["points"], errors="ignore").to_csv(outputs_dir / "training_segments.csv", index=False)
    phase_counts = features["phase"].value_counts().to_dict()
    save_json(phase_counts, outputs_dir / "phase_counts.json")
    manifest_path = Path(str(args.input) + ".manifest.json") if args.input else None
    source_manifest = None
    if manifest_path and manifest_path.exists():
        try:
            import json

            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            source_manifest = None
    save_json(
        {"models": summaries, "n_segments": int(len(features)), "training_source": training_source, "source_manifest": source_manifest},
        outputs_dir / "training_summary.json",
    )

    cluster_rows = []
    for item in summaries:
        cluster_rows.append(item)
    pd.DataFrame(cluster_rows).to_csv(outputs_dir / "cluster_summary.csv", index=False)
    LOGGER.info("Trained %s models from %s segments", len(summaries), len(features))
    return features


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train phase-specific ADS-B DBSCAN models")
    parser.add_argument("--input", help="CSV, Parquet, JSON, or JSONL ADS-B file")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default="artifacts/models")
    parser.add_argument("--input-radius-nm", type=float, default=150.0, help="When input is a readsb trace directory, keep points within this radius of the configured airport")
    parser.add_argument("--limit-files", type=int, default=None, help="Debug option: only scan the first N trace files")
    parser.add_argument("--max-segments-per-phase", type=int, default=5000, help="Cap segments fitted per phase so manual DBSCAN remains tractable")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    if not args.input:
        raise SystemExit("--input is required")
    run_training(args)


if __name__ == "__main__":
    main()
