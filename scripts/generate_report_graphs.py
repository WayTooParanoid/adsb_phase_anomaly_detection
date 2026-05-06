from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adsb_phase_dbscan.config import load_config
from adsb_phase_dbscan.model_store import load_models
from adsb_phase_dbscan.phase_classifier import classify_segments
from adsb_phase_dbscan.score_offline import score_features


PHASE_ORDER = [
    "INITIAL_APPROACH",
    "INTERMEDIATE_APPROACH",
    "FINAL_APPROACH",
    "DEPARTURE",
    "ENROUTE_CLIMB",
    "UNKNOWN",
]

PHASE_COLORS = {
    "INITIAL_APPROACH": "#2563eb",
    "INTERMEDIATE_APPROACH": "#0891b2",
    "FINAL_APPROACH": "#7c3aed",
    "DEPARTURE": "#16a34a",
    "ENROUTE_CLIMB": "#f97316",
    "UNKNOWN": "#6b7280",
    "BASELINE_GLOBAL_AVIATION": "#334155",
}


def _phase_sort(values: pd.Series | list[str]) -> list[str]:
    seen = [str(v) for v in values if pd.notna(v)]
    known = [phase for phase in PHASE_ORDER if phase in seen]
    extra = sorted({phase for phase in seen if phase not in PHASE_ORDER})
    return known + extra


def _colors(phases: list[str]) -> list[str]:
    return [PHASE_COLORS.get(phase, "#64748b") for phase in phases]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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


def plot_phase_distribution(segments: pd.DataFrame, out: Path) -> Path:
    counts = segments["phase"].astype(str).value_counts()
    phases = _phase_sort(counts.index.tolist())
    values = counts.reindex(phases).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(phases, values.values, color=_colors(phases))
    ax.set_title("Training Segment Distribution by Phase")
    ax.set_ylabel("Segments")
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    _bar_labels(ax, values.values.tolist(), "{:,.0f}")
    path = out / "phase_distribution.png"
    _save(fig, path)
    return path


def plot_daily_segments(segments: pd.DataFrame, out: Path) -> Path:
    daily = segments.groupby("source_day", dropna=False).size()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index.astype(str), daily.values, marker="o", color="#2563eb", linewidth=2)
    ax.fill_between(range(len(daily)), daily.values, alpha=0.12, color="#2563eb")
    ax.set_title("Daily Segment Volume")
    ax.set_ylabel("Segments")
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    path = out / "daily_segments.png"
    _save(fig, path)
    return path


def plot_daily_phase_heatmap(segments: pd.DataFrame, out: Path) -> Path:
    phases = _phase_sort(segments["phase"].astype(str).unique().tolist())
    table = pd.crosstab(segments["source_day"].astype(str), segments["phase"].astype(str)).reindex(columns=phases).fillna(0)
    row_pct = table.div(table.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(row_pct.to_numpy(dtype=float), aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_title("Daily Phase Mix (% of Segments)")
    ax.set_xticks(range(len(row_pct.columns)), row_pct.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(row_pct.index)), row_pct.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of day")
    path = out / "daily_phase_heatmap.png"
    _save(fig, path)
    return path


def plot_model_fit_coverage(summary: pd.DataFrame, out: Path) -> Path:
    data = summary[summary["phase"] != "BASELINE_GLOBAL_AVIATION"].copy()
    phases = _phase_sort(data["phase"].tolist())
    data = data.set_index("phase").reindex(phases)
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, data["n_clean_training_segments"], width, label="clean available", color="#94a3b8")
    ax.bar(x + width / 2, data["n_segments"], width, label="fit sample", color="#2563eb")
    ax.set_title("Model Fit Coverage")
    ax.set_ylabel("Segments")
    ax.set_xticks(x, data.index, rotation=35, ha="right")
    ax.legend(frameon=False)
    _format_axis(ax)
    path = out / "model_fit_coverage.png"
    _save(fig, path)
    return path


def plot_noise_rates(summary: pd.DataFrame, out: Path) -> Path:
    data = summary.copy()
    phases = _phase_sort(data["phase"].tolist())
    if "BASELINE_GLOBAL_AVIATION" in data["phase"].values:
        phases.append("BASELINE_GLOBAL_AVIATION")
        phases = list(dict.fromkeys(phases))
    data = data.set_index("phase").reindex(phases)
    values = data["noise_rate"].astype(float) * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(data.index, values, color=_colors(list(data.index)))
    ax.set_title("DBSCAN Training Noise Rate by Model")
    ax.set_ylabel("Noise rate (%)")
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    _bar_labels(ax, values.tolist(), "{:.1f}%")
    path = out / "dbscan_noise_rates.png"
    _save(fig, path)
    return path


def plot_cluster_counts(summary: pd.DataFrame, out: Path) -> Path:
    data = summary.copy()
    phases = _phase_sort(data["phase"].tolist())
    if "BASELINE_GLOBAL_AVIATION" in data["phase"].values:
        phases.append("BASELINE_GLOBAL_AVIATION")
        phases = list(dict.fromkeys(phases))
    data = data.set_index("phase").reindex(phases)
    values = data["n_clusters"].astype(float)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data.index, values, color=_colors(list(data.index)))
    ax.set_title("Learned Cluster Count by Model")
    ax.set_ylabel("Clusters")
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    _bar_labels(ax, values.tolist(), "{:.0f}")
    path = out / "dbscan_cluster_counts.png"
    _save(fig, path)
    return path


def plot_selected_eps(summary: pd.DataFrame, out: Path) -> Path:
    data = summary.copy()
    phases = _phase_sort(data["phase"].tolist())
    if "BASELINE_GLOBAL_AVIATION" in data["phase"].values:
        phases.append("BASELINE_GLOBAL_AVIATION")
        phases = list(dict.fromkeys(phases))
    data = data.set_index("phase").reindex(phases)
    values = data["selected_eps"].astype(float)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(data.index, values, color=_colors(list(data.index)))
    ax.set_title("Selected DBSCAN eps by Model")
    ax.set_ylabel("eps")
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    _bar_labels(ax, values.tolist(), "{:.2f}")
    path = out / "selected_eps_by_phase.png"
    _save(fig, path)
    return path


def plot_context_anomaly_rates(segments: pd.DataFrame, out: Path) -> Path:
    data = segments.copy()
    data["context_anomaly"] = data.get("context_anomaly", False).fillna(False).astype(bool)
    rates = data.groupby("phase")["context_anomaly"].mean() * 100
    phases = _phase_sort(rates.index.tolist())
    values = rates.reindex(phases).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(phases, values.values, color=_colors(phases))
    ax.set_title("Airport Context Anomaly Rate by Assigned Phase")
    ax.set_ylabel("Context anomaly rate (%)")
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    _bar_labels(ax, values.tolist(), "{:.1f}%")
    path = out / "context_anomaly_rates.png"
    _save(fig, path)
    return path


def plot_phase_confidence(segments: pd.DataFrame, out: Path) -> Path:
    phases = _phase_sort(segments["phase"].astype(str).unique().tolist())
    groups = [
        pd.to_numeric(segments.loc[segments["phase"].astype(str) == phase, "phase_confidence"], errors="coerce").dropna()
        for phase in phases
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    boxes = ax.boxplot(groups, patch_artist=True, tick_labels=phases, showfliers=False)
    for patch, color in zip(boxes["boxes"], _colors(phases)):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_title("Phase Assignment Confidence Distribution")
    ax.set_ylabel("Confidence")
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    path = out / "phase_confidence_distribution.png"
    _save(fig, path)
    return path


def plot_altitude_speed_scatter(segments: pd.DataFrame, out: Path, max_points: int = 8000) -> Path:
    data = segments.dropna(subset=["mean_ground_speed_kt", "mean_altitude_agl_ft", "phase"]).copy()
    if len(data) > max_points:
        sample_size = max(50, max_points // max(data["phase"].nunique(), 1))
        data = pd.concat(
            [group.sample(min(len(group), sample_size), random_state=7) for _, group in data.groupby("phase")],
            ignore_index=True,
        )
    fig, ax = plt.subplots(figsize=(11, 7))
    for phase in _phase_sort(data["phase"].astype(str).unique().tolist()):
        group = data[data["phase"].astype(str) == phase]
        ax.scatter(
            group["mean_ground_speed_kt"],
            group["mean_altitude_agl_ft"],
            s=8,
            alpha=0.35,
            label=phase,
            color=PHASE_COLORS.get(phase, "#64748b"),
        )
    ax.set_title("Speed vs Altitude by Assigned Phase")
    ax.set_xlabel("Mean ground speed (kt)")
    ax.set_ylabel("Mean altitude AGL (ft)")
    ax.legend(frameon=False, markerscale=2, fontsize=8, ncol=2)
    _format_axis(ax)
    path = out / "altitude_speed_by_phase.png"
    _save(fig, path)
    return path


def plot_feature_profiles(segments: pd.DataFrame, out: Path) -> Path:
    features = ["mean_altitude_agl_ft", "mean_ground_speed_kt", "mean_vertical_rate_fpm", "heading_change_deg", "sinuosity"]
    phases = _phase_sort(segments["phase"].astype(str).unique().tolist())
    medians = segments.groupby("phase")[features].median(numeric_only=True).reindex(phases)
    scaled = medians.copy()
    for col in features:
        series = scaled[col].astype(float)
        spread = series.quantile(0.75) - series.quantile(0.25)
        scaled[col] = (series - series.median()) / (spread if spread else 1.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(scaled.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_title("Median Aviation Feature Profile by Phase")
    ax.set_xticks(range(len(features)), [name.replace("_", " ") for name in features], rotation=30, ha="right")
    ax.set_yticks(range(len(phases)), phases)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Robust-scaled median")
    path = out / "feature_profiles_by_phase.png"
    _save(fig, path)
    return path


def plot_airport_context_distribution(segments: pd.DataFrame, out: Path) -> Path:
    top = segments["airport_id"].fillna("").replace("", "NO_CONTEXT").value_counts().head(12)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(top.index.astype(str), top.values, color="#2563eb")
    ax.set_title("Top Airport Context Assignments")
    ax.set_ylabel("Segments")
    ax.tick_params(axis="x", rotation=35)
    _format_axis(ax)
    _bar_labels(ax, top.values.tolist(), "{:,.0f}")
    path = out / "airport_context_distribution.png"
    _save(fig, path)
    return path


def build_best_model_heatmap(
    segments: pd.DataFrame,
    models_dir: Path,
    config_path: Path | None,
    out: Path,
    sample_per_phase: int,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    models = load_models(models_dir)
    sample = pd.concat(
        [group.sample(min(len(group), sample_per_phase), random_state=11) for _, group in segments.groupby("phase")],
        ignore_index=True,
    )
    scored = score_features(sample, models, config)
    scored = scored[scored["model_best_phase"].notna()].copy()
    phases = _phase_sort(scored["phase"].astype(str).unique().tolist())
    best_phases = _phase_sort(scored["model_best_phase"].astype(str).unique().tolist())
    counts = pd.crosstab(scored["phase"].astype(str), scored["model_best_phase"].astype(str)).reindex(index=phases, columns=best_phases).fillna(0).astype(int)
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    counts.to_csv(out / "best_model_vs_phase_assignment_counts.csv")
    pct.round(2).to_csv(out / "best_model_vs_phase_assignment_percent.csv")
    scored[
        [
            "segment_id",
            "phase",
            "model_best_phase",
            "model_best_score",
            "model_best_cluster",
            "anomaly_score",
            "is_anomaly",
        ]
    ].to_csv(out / "best_model_vs_phase_assignment_sample.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(pct.fillna(0).to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=100)
    ax.set_title("Best DBSCAN Model Fit vs Rule-Based Phase Assignment")
    ax.set_xlabel("Best-fitting DBSCAN model")
    ax.set_ylabel("Assigned phase")
    ax.set_xticks(range(len(best_phases)), best_phases, rotation=35, ha="right")
    ax.set_yticks(range(len(phases)), phases)
    for i in range(len(phases)):
        for j in range(len(best_phases)):
            value = pct.iloc[i, j]
            if pd.notna(value) and value >= 0.5:
                ax.text(j, i, f"{value:.0f}%", ha="center", va="center", color="white" if value > 55 else "#0f172a", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% of assigned phase sample")
    path = out / "best_model_vs_phase_assignment.png"
    _save(fig, path)
    return path, counts, pct, scored


def plot_evaluation_metrics(
    segments: pd.DataFrame,
    summary: pd.DataFrame,
    best_counts: pd.DataFrame,
    best_pct: pd.DataFrame,
    out: Path,
) -> Path:
    non_unknown = segments["phase"].astype(str) != "UNKNOWN"
    unknown_rate = 100 * (1 - non_unknown.mean())
    context_rate = 100 * segments.get("context_anomaly", pd.Series(False, index=segments.index)).fillna(False).astype(bool).mean()
    model_phases = set(summary.loc[summary["phase"] != "BASELINE_GLOBAL_AVIATION", "phase"].astype(str))
    model_coverage = 100 * segments["phase"].astype(str).isin(model_phases).mean()
    diag = 0
    total = int(best_counts.to_numpy().sum()) if not best_counts.empty else 0
    for phase in best_counts.index:
        if phase in best_counts.columns:
            diag += int(best_counts.loc[phase, phase])
    best_match = 100 * diag / total if total else 0.0
    known_best = best_counts.loc[[idx for idx in best_counts.index if idx != "UNKNOWN"]]
    known_total = int(known_best.to_numpy().sum()) if not known_best.empty else 0
    known_diag = sum(int(known_best.loc[phase, phase]) for phase in known_best.index if phase in known_best.columns)
    known_best_match = 100 * known_diag / known_total if known_total else 0.0
    mean_noise = 100 * summary.loc[summary["phase"] != "BASELINE_GLOBAL_AVIATION", "noise_rate"].astype(float).mean()

    labels = [
        "model coverage",
        "known phase rate",
        "best-fit match",
        "known best-fit match",
        "mean DBSCAN noise",
        "context anomaly rate",
    ]
    values = [model_coverage, 100 - unknown_rate, best_match, known_best_match, mean_noise, context_rate]
    colors = ["#2563eb", "#16a34a", "#0891b2", "#14b8a6", "#f97316", "#dc2626"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, values, color=colors)
    ax.set_title("Unsupervised Evaluation Summary")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, max(100, max(values) * 1.15))
    ax.tick_params(axis="x", rotation=25)
    _format_axis(ax)
    _bar_labels(ax, values, "{:.1f}%")
    path = out / "evaluation_performance_metrics.png"
    _save(fig, path)
    return path


def load_summary(cluster_summary: Path, training_summary: Path | None) -> pd.DataFrame:
    if cluster_summary.exists():
        return pd.read_csv(cluster_summary)
    if training_summary and training_summary.exists():
        with training_summary.open("r", encoding="utf-8") as fh:
            return pd.DataFrame(json.load(fh)["models"])
    raise FileNotFoundError("No cluster summary or training summary was found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report graphs for ADS-B Phase DBSCAN")
    parser.add_argument("--segments", default="artifacts/training_cache/atl_segments.csv.gz")
    parser.add_argument("--cluster-summary", default="artifacts/outputs/cluster_summary.csv")
    parser.add_argument("--training-summary", default="artifacts/outputs/training_summary.json")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--config", default="config.real.yaml")
    parser.add_argument("--output-dir", default="report_assets")
    parser.add_argument("--sample-per-phase", type=int, default=500)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    segments = pd.read_csv(args.segments, low_memory=False)
    segments = classify_segments(segments, config)
    summary = load_summary(Path(args.cluster_summary), Path(args.training_summary))
    generated: list[Path] = []
    generated.append(plot_phase_distribution(segments, out))
    generated.append(plot_daily_segments(segments, out))
    generated.append(plot_daily_phase_heatmap(segments, out))
    generated.append(plot_model_fit_coverage(summary, out))
    generated.append(plot_noise_rates(summary, out))
    generated.append(plot_cluster_counts(summary, out))
    generated.append(plot_selected_eps(summary, out))
    generated.append(plot_context_anomaly_rates(segments, out))
    generated.append(plot_phase_confidence(segments, out))
    generated.append(plot_altitude_speed_scatter(segments, out))
    generated.append(plot_feature_profiles(segments, out))
    generated.append(plot_airport_context_distribution(segments, out))
    heatmap_path, counts, pct, _ = build_best_model_heatmap(
        segments,
        Path(args.models_dir),
        Path(args.config) if args.config else None,
        out,
        args.sample_per_phase,
    )
    generated.append(heatmap_path)
    generated.append(plot_evaluation_metrics(segments, summary, counts, pct, out))

    graph_index = {
        "generated": [str(path) for path in generated],
        "inputs": {
            "segments": str(Path(args.segments)),
            "cluster_summary": str(Path(args.cluster_summary)),
            "models_dir": str(Path(args.models_dir)),
            "config": args.config,
            "sample_per_phase": args.sample_per_phase,
        },
    }
    (out / "graph_index.json").write_text(json.dumps(graph_index, indent=2), encoding="utf-8")
    print(f"Generated {len(generated)} graphs in {out}")


if __name__ == "__main__":
    main()
