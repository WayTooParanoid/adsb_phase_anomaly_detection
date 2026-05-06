# ADS-B Phase DBSCAN

This project builds phase-specific DBSCAN anomaly models for ADS-B trajectory segments around ATL. The runtime path is intentionally small:

1. Convert processed ADS-B day files into a segment feature cache.
2. Train one DBSCAN model per flight phase.
3. Score live or replay aircraft with the trained models.
4. Generate report figures from the retained cache and model outputs.

The active phases are arrival, departure, and related climb/approach behavior used by the current classifier/model scoring flow.

## Install

```bash
cd adsb_phase_dbscan
python -m pip install -e .
```

For Parquet cache files, install:

```bash
python -m pip install -e .[parquet]
```

## Build Training Cache

The normal ATL workflow starts from processed day files under `..\processed\extracted_days`:

```bash
build-training-cache.cmd
```

This writes:

- `artifacts/training_cache/atl_segments.csv.gz`
- `artifacts/training_cache/atl_segments.csv.gz.manifest.json`

The Python equivalent is:

```bash
python -m adsb_phase_dbscan.build_training_cache ^
  --input ..\processed\extracted_days ^
  --output artifacts\training_cache\atl_segments.csv.gz ^
  --config config.real.yaml ^
  --input-radius-nm 120
```

## Train Models

```bash
train-models.cmd
```

Training reads the feature cache, refreshes the phase labels with the current classifier, fits phase-specific DBSCAN models, and writes:

- `artifacts/models/*_dbscan.joblib`
- `artifacts/outputs/training_segments.csv`
- `artifacts/outputs/training_summary.json`
- `artifacts/outputs/phase_counts.json`
- `artifacts/outputs/cluster_summary.csv`

Each trained model also stores a per-phase nearest-core distance calibration. During scoring, the raw DBSCAN eps score is paired with a phase percentile such as `p98`, meaning the segment is farther from normal than 98% of that phase's training inliers.

## Live Dashboard

```bash
start_dashboard.cmd
```

The launcher opens `http://127.0.0.1:8050` after the server starts.

The dashboard shows live/replay aircraft, phase assignments, model scores, nearest airport markers, trails, and anomaly/watch state.

## Offline Scoring

```bash
python -m adsb_phase_dbscan.score_offline ^
  --input path\to\adsb.csv ^
  --models-dir artifacts\models ^
  --config config.real.yaml ^
  --output outputs\scored_segments.csv
```

## Replay Cache

Replay data for the dashboard is built with:

```bash
build-replay-cache.cmd
```

## Report Figures

Regenerate the report figures from the current cache and model outputs:

```bash
python scripts/generate_report_graphs.py
python scripts/generate_model_first_eval_graphs.py
```

Updated figures and CSV summaries are written under `report_assets/`.

## Required Code Layout

- `src/adsb_phase_dbscan/build_training_cache.py`: processed day files to training feature cache.
- `src/adsb_phase_dbscan/train.py`: phase-specific DBSCAN training.
- `src/adsb_phase_dbscan/score_offline.py`: batch scoring.
- `src/adsb_phase_dbscan/live_dashboard.py`: browser dashboard.
- `src/adsb_phase_dbscan/live_adsb_lol.py`: live ADS-B polling/scoring.
- `src/adsb_phase_dbscan/build_replay_cache.py`: replay cache creation.
- `src/adsb_phase_dbscan/features.py`, `phase_classifier.py`, `model_scoring.py`, `model_store.py`, `custom_dbscan.py`: core model logic.
- `src/adsb_phase_dbscan/airport_context.py`, `cleaning.py`, `track_builder.py`, `geometry.py`, `io.py`, `config.py`: data preparation and utilities required by the runtime path.
- `scripts/`: report figure generation only.
- `commands/` and root `.cmd` files: Windows launchers for the required workflow.

## Notes

- DBSCAN is used for anomaly scoring, not as the only source of flight phase truth.
- Airport-relative fields are used for context and dashboard display; the DBSCAN feature matrix stays aviation-behavior focused.
- The project is unsupervised. Without manually labeled anomalies, report outputs should be treated as model behavior summaries rather than ground-truth accuracy claims.
