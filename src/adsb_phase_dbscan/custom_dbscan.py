from __future__ import annotations

from collections import deque

import numpy as np

UNVISITED = -99
NOISE = -1


class CustomDBSCAN:
    def __init__(
        self,
        eps: float = 1.0,
        min_samples: int = 10,
        metric: str = "euclidean",
        batch_size: int = 1024,
        min_unique_tracks: int | None = None,
        feature_weights=None,
    ):
        if float(eps) <= 0:
            raise ValueError("eps must be positive")
        if int(min_samples) < 1:
            raise ValueError("min_samples must be at least 1")
        if metric != "euclidean":
            raise ValueError("Only euclidean metric is implemented")
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.metric = metric
        self.batch_size = max(1, int(batch_size))
        self.min_unique_tracks = None if min_unique_tracks is None else int(min_unique_tracks)
        if self.min_unique_tracks is not None and self.min_unique_tracks < 1:
            raise ValueError("min_unique_tracks must be at least 1 when provided")
        self.feature_weights = None if feature_weights is None else self._validate_feature_weights(feature_weights)

    @staticmethod
    def _validate_feature_weights(feature_weights, expected_features: int | None = None) -> np.ndarray:
        weights = np.asarray(feature_weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("feature_weights must be a 1D numeric array")
        if expected_features is not None and len(weights) != expected_features:
            raise ValueError(f"feature_weights has length {len(weights)}; expected {expected_features}")
        if len(weights) < 1:
            raise ValueError("feature_weights must contain at least one value")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("feature_weights must contain only positive finite values")
        return weights

    def _validate_X(self, X, *, name: str = "X", allow_empty: bool = False, expected_features: int | None = None):
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D numeric array")
        if arr.shape[1] < 1:
            raise ValueError(f"{name} must contain at least one feature")
        if not allow_empty and arr.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one sample")
        if expected_features is not None and arr.shape[1] != expected_features:
            raise ValueError(f"{name} has {arr.shape[1]} features; expected {expected_features}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must contain only finite values")
        feature_weights = getattr(self, "feature_weights", None)
        if feature_weights is not None:
            self._validate_feature_weights(feature_weights, arr.shape[1])
        return arr

    def _validate_group_ids(self, group_ids, n_samples: int):
        if self.min_unique_tracks is None or self.min_unique_tracks <= 1:
            return None
        if group_ids is None:
            raise ValueError("group_ids are required when min_unique_tracks is greater than 1")
        ids = np.asarray(group_ids, dtype=object)
        if ids.ndim != 1 or len(ids) != n_samples:
            raise ValueError(f"group_ids must be a 1D array with {n_samples} values")
        if any(item is None or str(item) == "" for item in ids):
            raise ValueError("group_ids must not contain empty values")
        return ids

    def _check_fitted(self) -> None:
        if not hasattr(self, "components_"):
            raise RuntimeError("CustomDBSCAN model is not fitted")

    def fit(self, X, group_ids=None):
        X = self._validate_X(X)
        group_ids = self._validate_group_ids(group_ids, X.shape[0])
        self.n_features_in_ = int(X.shape[1])
        labels = np.full(X.shape[0], UNVISITED, dtype=int)
        cluster_id = 0
        core_indices: list[int] = []
        for point_idx in range(X.shape[0]):
            if labels[point_idx] != UNVISITED:
                continue
            neighbors = self._region_query(X, point_idx)
            if not self._is_core(neighbors, group_ids):
                labels[point_idx] = NOISE
                continue
            core_indices.append(point_idx)
            self._expand_cluster(X, labels, point_idx, neighbors, cluster_id, core_indices, group_ids)
            cluster_id += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.array(sorted(set(core_indices)), dtype=int)
        self.components_ = X[self.core_sample_indices_] if len(self.core_sample_indices_) else np.empty((0, X.shape[1]))
        self.core_labels_ = labels[self.core_sample_indices_] if len(self.core_sample_indices_) else np.array([], dtype=int)
        return self

    def fit_predict(self, X, group_ids=None):
        return self.fit(X, group_ids=group_ids).labels_

    def _distances_to_point(self, X, point):
        diff = X - point
        feature_weights = getattr(self, "feature_weights", None)
        if feature_weights is None:
            return np.linalg.norm(diff, axis=1)
        return np.sqrt(np.sum(feature_weights * diff * diff, axis=1))

    def _region_query(self, X, point_idx: int) -> np.ndarray:
        distances = self._distances_to_point(X, X[point_idx])
        return np.flatnonzero(distances <= self.eps)

    def _is_core(self, neighbors, group_ids) -> bool:
        if len(neighbors) < self.min_samples:
            return False
        if group_ids is None or self.min_unique_tracks is None or self.min_unique_tracks <= 1:
            return True
        unique_tracks = {str(group_ids[int(idx)]) for idx in neighbors}
        return len(unique_tracks) >= self.min_unique_tracks

    def _expand_cluster(self, X, labels, point_idx: int, neighbors, cluster_id: int, core_indices: list[int], group_ids) -> None:
        labels[point_idx] = cluster_id
        queue = deque(int(n) for n in neighbors)
        seen = set(queue)
        while queue:
            neighbor_idx = queue.popleft()
            if labels[neighbor_idx] == NOISE:
                labels[neighbor_idx] = cluster_id
            if labels[neighbor_idx] != UNVISITED:
                continue
            labels[neighbor_idx] = cluster_id
            neighbor_neighbors = self._region_query(X, neighbor_idx)
            if self._is_core(neighbor_neighbors, group_ids):
                core_indices.append(neighbor_idx)
                for candidate in neighbor_neighbors:
                    candidate = int(candidate)
                    if candidate not in seen:
                        queue.append(candidate)
                        seen.add(candidate)

    def _nearest_core(self, X_new):
        self._check_fitted()
        X_new = self._validate_X(X_new, name="X_new", expected_features=self.n_features_in_)
        if len(self.components_) == 0:
            distances = np.full(X_new.shape[0], np.nan)
            indices = np.full(X_new.shape[0], -1, dtype=int)
            return distances, indices
        out_distances = np.empty(X_new.shape[0], dtype=float)
        out_indices = np.empty(X_new.shape[0], dtype=int)
        for start in range(0, X_new.shape[0], self.batch_size):
            stop = min(start + self.batch_size, X_new.shape[0])
            batch = X_new[start:stop]
            diff = batch[:, None, :] - self.components_[None, :, :]
            feature_weights = getattr(self, "feature_weights", None)
            if feature_weights is None:
                distances = np.linalg.norm(diff, axis=2)
            else:
                distances = np.sqrt(np.sum(feature_weights * diff * diff, axis=2))
            indices = np.argmin(distances, axis=1)
            out_distances[start:stop] = distances[np.arange(len(batch)), indices]
            out_indices[start:stop] = indices
        return out_distances, out_indices

    def predict_from_core_samples(self, X_new):
        nearest_dist, nearest_idx = self._nearest_core(X_new)
        labels = np.full(len(nearest_dist), NOISE, dtype=int)
        mask = nearest_dist <= self.eps
        if len(self.core_labels_):
            labels[mask] = self.core_labels_[nearest_idx[mask]]
        return labels

    def anomaly_score(self, X_new):
        nearest_dist, _ = self._nearest_core(X_new)
        return nearest_dist / max(self.eps, 1e-9)

    def nearest_core_distance(self, X_new):
        nearest_dist, _ = self._nearest_core(X_new)
        return nearest_dist

    def nearest_core_feature_contributions(self, X_new, feature_names: list[str] | None = None, top_n: int = 3):
        nearest_dist, nearest_idx = self._nearest_core(X_new)
        X_new = self._validate_X(X_new, name="X_new", expected_features=self.n_features_in_)
        if feature_names is None:
            feature_names = [f"feature_{idx}" for idx in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError(f"feature_names has length {len(feature_names)}; expected {self.n_features_in_}")
        top_n = max(1, int(top_n))
        feature_weights = getattr(self, "feature_weights", None)
        out = []
        for row_idx, core_idx in enumerate(nearest_idx):
            if core_idx < 0 or len(self.components_) == 0:
                out.append([])
                continue
            diff = X_new[row_idx] - self.components_[core_idx]
            weighted_sq = diff * diff if feature_weights is None else feature_weights * diff * diff
            order = np.argsort(weighted_sq)[::-1]
            items = []
            for idx in order[:top_n]:
                idx = int(idx)
                contribution = float(weighted_sq[idx])
                if contribution <= 0:
                    continue
                items.append(
                    {
                        "feature": feature_names[idx],
                        "normalized_delta": float(diff[idx]),
                        "contribution": contribution,
                    }
                )
            out.append(items)
        return out
