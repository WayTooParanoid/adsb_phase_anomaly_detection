# Phase-Aware ADS-B DBSCAN

A phase-specific unsupervised learning system for analyzing ADS-B flight behavior near Hartsfield-Jackson Atlanta International Airport (ATL).

## Overview

This project analyzes ADS-B trajectory behavior around ATL using a phase-aware DBSCAN workflow. The main idea is that flight behavior is highly dependent on operational context. A speed, altitude, descent rate, turn, or path shape that is normal during one phase of flight may be unusual during another.

Instead of using one global anomaly detector, the system trains separate DBSCAN models for different flight phases:

- `DEPARTURE`
- `ENROUTE_CLIMB`
- `INITIAL_APPROACH`
- `INTERMEDIATE_APPROACH`
- `FINAL_APPROACH`
- `UNKNOWN`

Each phase model learns its own dense region of normal behavior. At runtime, every trained model scores the same trajectory segment, and the phase with the lowest finite raw DBSCAN distance is selected.

The system then reports whether the segment fits the selected phase model or appears behaviorally anomalous.

## Key Features

- Phase-specific DBSCAN models for ADS-B behavior analysis
- No need for labeled anomaly examples
- Model-first phase selection at runtime
- Learned `UNKNOWN` class for ambiguous or non-ATL-focused traffic
- Raw DBSCAN anomaly score based on nearest-core distance
- Feature-level explanations for anomaly results
- Dashboard support for live and replay-based testing
- Balanced weak-label evaluation to avoid UNKNOWN dominating the metric

## Dataset

Training uses processed ADS-B trajectory segments collected around ATL from **December 20-30, 2025**.

After cleaning and segment construction, the retained cache contains:

```text
428,212 trajectory segments
