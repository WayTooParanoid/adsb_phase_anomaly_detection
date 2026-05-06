from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adsb_phase_dbscan.config import load_config
from adsb_phase_dbscan.model_store import load_models
from adsb_phase_dbscan.score_offline import score_features


PHASE_COLORS = {
    "INITIAL_APPROACH": "#2563eb",
    "INTERMEDIATE_APPROACH": "#0891b2",
    "FINAL_APPROACH": "#7c3aed",
    "DEPARTURE": "#16a34a",
    "ENROUTE_CLIMB": "#ea580c",
    "UNKNOWN": "#475569",
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _format_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bar_labels(ax: plt.Axes, values: list[float], fmt: str = "{:.0f}") -> None:
    for patch, value in zip(ax.patches, values):
        if not np.isfinite(value):
            continue
        ax.annotate(
            fmt.format(value),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )


def _phase_order(phases: list[str]) -> list[str]:
    preferred = ["INITIAL_APPROACH", "INTERMEDIATE_APPROACH", "FINAL_APPROACH", "DEPARTURE", "ENROUTE_CLIMB", "UNKNOWN"]
    return [phase for phase in preferred if phase in phases] + sorted(set(phases) - set(preferred))


def _colors(phases: list[str]) -> list[str]:
    return [PHASE_COLORS.get(phase, "#64748b") for phase in phases]


def _sample_segments(segments: pd.DataFrame, sample_per_phase: int, max_rows: int) -> pd.DataFrame:
    if sample_per_phase <= 0 and max_rows <= 0:
        return segments.copy()
    rows = []
    if sample_per_phase > 0 and "phase" in segments.columns:
        for _, group in segments.groupby("phase", dropna=False):
            rows.append(group.sample(min(len(group), sample_per_phase), random_state=17))
        sampled = pd.concat(rows, ignore_index=True) if rows else segments.head(0).copy()
    else:
        sampled = segments.copy()
    if max_rows > 0 and len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=19).reset_index(drop=True)
    return sampled.reset_index(drop=True)


def _contribution_counter(scored: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    total_delta: Counter[str] = Counter()
    anomaly_rows = scored[scored.get("behavior_anomaly", False).fillna(False).astype(bool)]
    for contributions in anomaly_rows.get("dbscan_feature_contributions", []):
        if not isinstance(contributions, list):
            continue
        for item in contributions[:3]:
            if not isinstance(item, dict):
                continue
            feature = str(item.get("feature") or "")
            if not feature:
                continue
            try:
                delta = abs(float(item.get("normalized_delta", 0.0)))
            except (TypeError, ValueError):
                delta = 0.0
            counter[feature] += 1
            total_delta[feature] += delta
    rows = [
        {
            "feature": feature,
            "count": count,
            "mean_abs_normalized_delta": total_delta[feature] / count if count else 0.0,
        }
        for feature, count in counter.most_common()
    ]
    return pd.DataFrame(rows)


def plot_phase_counts(scored: pd.DataFrame, out: Path) -> Path:
    counts = scored["phase"].astype(str).value_counts()
    phases = _phase_order(counts.index.tolist())
    values = counts.reindex(phases).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(phases, values.values, color=_colors(phases))
    ax.set_title("Model-First Assigned Phase Counts")
    ax.set_ylabel("Segments")
    ax.tick_params(axis="x", rotation=30)
    _format_axis(ax)
    _bar_labels(ax, values.values.tolist(), "{:,.0f}")
    return _save(fig, out / "model_first_phase_counts.png")


def plot_anomaly_rates(scored: pd.DataFrame, out: Path) -> Path:
    data = scored.copy()
    data["behavior_anomaly"] = data["behavior_anomaly"].fillna(False).astype(bool)
    rates = data.groupby("phase").agg(segments=("phase", "size"), anomaly_rate=("behavior_anomaly", "mean")).reset_index()
    rates["phase"] = rates["phase"].astype(str)
    phases = _phase_order(rates["phase"].tolist())
    rates = rates.set_index("phase").reindex(phases).reset_index()
    rates.to_csv(out / "model_first_anomaly_rates.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(10.5, 5.5))
    ax1.bar(rates["phase"], rates["anomaly_rate"], color=_colors(rates["phase"].tolist()), alpha=0.9)
    ax1.set_title("Model-First Behavior Anomaly Rate by Assigned Phase")
    ax1.set_ylabel("Behavior anomaly rate")
    ax1.set_ylim(0, max(0.05, float(rates["anomaly_rate"].max()) * 1.25))
    ax1.tick_params(axis="x", rotation=30)
    _format_axis(ax1)
    _bar_labels(ax1, rates["anomaly_rate"].fillna(0).tolist(), "{:.1%}")
    ax2 = ax1.twinx()
    ax2.plot(rates["phase"], rates["segments"], color="#111827", marker="o", linewidth=1.4)
    ax2.set_ylabel("Segments")
    return _save(fig, out / "model_first_anomaly_rates.png")


def plot_score_distribution(scored: pd.DataFrame, out: Path) -> Path:
    data = scored.copy()
    data["anomaly_score"] = pd.to_numeric(data["anomaly_score"], errors="coerce")
    data = data.dropna(subset=["anomaly_score"])
    phases = _phase_order(data["phase"].astype(str).unique().tolist())
    clipped = data["anomaly_score"].clip(upper=data["anomaly_score"].quantile(0.99))
    data = data.assign(_score_clipped=clipped)
    values = [data.loc[data["phase"].astype(str) == phase, "_score_clipped"].to_numpy() for phase in phases]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.boxplot(values, tick_labels=phases, showfliers=False)
    ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="DBSCAN core boundary")
    ax.set_title("Model-First Anomaly Score Distribution by Phase")
    ax.set_ylabel("Anomaly score (clipped at p99)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False)
    _format_axis(ax)
    return _save(fig, out / "model_first_score_distribution.png")


def plot_severity_counts(scored: pd.DataFrame, out: Path) -> Path:
    severity_order = ["NORMAL", "EDGE", "UNUSUAL", "OUTLIER", "NO_MODEL", "NO_CORE"]
    counts = scored["model_severity"].fillna("NO_MODEL").astype(str).value_counts().reindex(severity_order).fillna(0).astype(int)
    colors = ["#16a34a", "#facc15", "#f97316", "#dc2626", "#94a3b8", "#64748b"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Model Severity Mix")
    ax.set_ylabel("Segments")
    _format_axis(ax)
    _bar_labels(ax, counts.values.tolist(), "{:,.0f}")
    return _save(fig, out / "model_first_severity_counts.png")


def plot_assignment_consistency(scored: pd.DataFrame, out: Path) -> tuple[Path, float]:
    data = scored[scored["model_best_phase"].notna()].copy()
    assigned = _phase_order(data["phase"].astype(str).unique().tolist())
    best = _phase_order(data["model_best_phase"].astype(str).unique().tolist())
    counts = (
        pd.crosstab(data["phase"].astype(str), data["model_best_phase"].astype(str))
        .reindex(index=assigned, columns=best)
        .fillna(0)
        .astype(int)
    )
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    counts.to_csv(out / "model_first_assignment_consistency_counts.csv")
    pct.round(2).to_csv(out / "model_first_assignment_consistency_percent.csv")
    diagonal = sum(int(counts.loc[phase, phase]) for phase in counts.index if phase in counts.columns)
    total = int(counts.to_numpy().sum())
    consistency = diagonal / total if total else 0.0

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pct.fillna(0).to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=100)
    ax.set_title("Model-First Assigned Phase vs Best-Fitting DBSCAN Model")
    ax.set_xlabel("Best-fitting DBSCAN model")
    ax.set_ylabel("Assigned phase")
    ax.set_xticks(range(len(counts.columns)), counts.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(counts.index)), counts.index)
    for i in range(len(counts.index)):
        for j in range(len(counts.columns)):
            value = pct.iloc[i, j]
            count = int(counts.iloc[i, j])
            if count and pd.notna(value):
                ax.text(j, i, f"{value:.0f}%\n{count}", ha="center", va="center", color="white" if value > 55 else "#0f172a", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of assigned phase")
    return _save(fig, out / "model_first_assignment_consistency_heatmap.png"), consistency


def plot_top_features(features: pd.DataFrame, out: Path) -> Path | None:
    if features.empty:
        return None
    top = features.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top["feature"], top["count"], color="#7c3aed")
    ax.set_title("Most Common Feature Drivers Among Behavior Anomalies")
    ax.set_ylabel("Top-3 contribution appearances")
    ax.tick_params(axis="x", rotation=30)
    _format_axis(ax)
    _bar_labels(ax, top["count"].tolist(), "{:,.0f}")
    return _save(fig, out / "model_first_top_anomaly_features.png")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model-first scoring evaluation visuals")
    parser.add_argument("--segments", default="artifacts/training_cache/atl_segments.csv.gz")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--config", default="config.real.yaml")
    parser.add_argument("--output-dir", default="artifacts/model_first_eval")
    parser.add_argument("--sample-per-phase", type=int, default=8000)
    parser.add_argument("--max-rows", type=int, default=50000)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    models = load_models(args.models_dir)
    segments = pd.read_csv(args.segments, low_memory=False)
    sample = _sample_segments(segments, args.sample_per_phase, args.max_rows)
    scored = score_features(sample, models, config, assign_phase_from_models=True)
    scored.drop(columns=["points"], errors="ignore").to_csv(out / "model_first_scored_sample.csv", index=False)

    top_anomalies = scored.sort_values("anomaly_score", ascending=False, na_position="last").head(100)
    top_anomalies.drop(columns=["points"], errors="ignore").to_csv(out / "model_first_top_anomalies.csv", index=False)
    feature_drivers = _contribution_counter(scored)
    feature_drivers.to_csv(out / "model_first_top_anomaly_features.csv", index=False)

    generated: list[Path] = []
    generated.append(plot_phase_counts(scored, out))
    generated.append(plot_anomaly_rates(scored, out))
    generated.append(plot_score_distribution(scored, out))
    generated.append(plot_severity_counts(scored, out))
    assignment_heatmap, assignment_consistency = plot_assignment_consistency(scored, out)
    generated.append(assignment_heatmap)
    top_feature_plot = plot_top_features(feature_drivers, out)
    if top_feature_plot is not None:
        generated.append(top_feature_plot)

    summary = {
        "segments": str(Path(args.segments)),
        "models_dir": str(Path(args.models_dir)),
        "config": args.config,
        "sample_rows": int(len(scored)),
        "models_loaded": sorted(models.keys()),
        "assigned_phase_counts": {str(k): int(v) for k, v in scored["phase"].astype(str).value_counts().to_dict().items()},
        "unknown_rate": float((scored["phase"].astype(str) == "UNKNOWN").mean()),
        "behavior_anomaly_rate": float(scored["behavior_anomaly"].fillna(False).astype(bool).mean()),
        "combined_anomaly_rate": float(scored["is_anomaly"].fillna(False).astype(bool).mean()),
        "assignment_best_model_consistency": float(assignment_consistency),
        "median_anomaly_score": float(pd.to_numeric(scored["anomaly_score"], errors="coerce").median(skipna=True)),
        "p95_anomaly_score": float(pd.to_numeric(scored["anomaly_score"], errors="coerce").quantile(0.95)),
        "generated": [str(path) for path in generated],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8")
    print(f"Generated {len(generated)} model-first evaluation graphs in {out}")


if __name__ == "__main__":
    main()
