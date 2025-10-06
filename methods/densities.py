# densities.py
"""Density-based clustering and kernel density estimation.

Provides a performant DBSCAN using a KD-tree for neighbor queries, a thin wrapper for
OPTICS that defers to sklearn when available, and a small KDE wrapper using scipy.stats.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY_STATS = True
except Exception:
    _HAS_SCIPY_STATS = False

try:
    from sklearn.cluster import OPTICS as _SklearnOPTICS
    _HAS_SKLEARN_OPTICS = True
except Exception:
    _HAS_SKLEARN_OPTICS = False

logger = logging.getLogger(__name__)


class DBSCAN:
    """DBSCAN implementation using a KD-tree for faster neighborhood queries on large datasets.

    This implementation mirrors sklearn's behavior for labels: -1 indicates noise.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.labels_: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X)
        n_samples = X.shape[0]
        labels = -np.ones(n_samples, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        tree = cKDTree(X)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = tree.query_ball_point(X[i], self.eps)

            if len(neighbors) < self.min_samples:
                # Noise for now; can be later assigned
                labels[i] = -1
            else:
                self._expand_cluster(tree, X, labels, visited, i, neighbors, cluster_id)
                cluster_id += 1

        self.labels_ = labels
        self._is_fitted = True
        logger.info("DBSCAN found %d clusters (excluding noise)", cluster_id)
        return self

    def _expand_cluster(self, tree: cKDTree, X: np.ndarray, labels: np.ndarray, visited: np.ndarray, point_idx: int, neighbors: list, cluster_id: int) -> None:
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                new_neighbors = tree.query_ball_point(X[neighbor_idx], self.eps)
                if len(new_neighbors) >= self.min_samples:
                    # Append only new indices to avoid explosion
                    neighbors += [n for n in new_neighbors if n not in neighbors]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            i += 1


class OPTICS:
    """Thin wrapper that uses sklearn.cluster.OPTICS when available.

    If sklearn's OPTICS is unavailable, this class raises ImportError and suggests installing scikit-learn.
    """

    def __init__(self, min_samples: int = 5, max_eps: float = np.inf, xi: float = 0.05, min_cluster_size: float = 0.1):
        if not _HAS_SKLEARN_OPTICS:
            raise ImportError("sklearn.cluster.OPTICS is required for OPTICS wrapper. Install scikit-learn >= 0.21.")
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self._model = _SklearnOPTICS(min_samples=self.min_samples, max_eps=self.max_eps, xi=self.xi, min_cluster_size=self.min_cluster_size)

    def fit(self, X: np.ndarray):
        self._model.fit(X)
        self.reachability_ = getattr(self._model, "reachability_")
        self.ordering_ = getattr(self._model, "ordering_")
        self.labels_ = getattr(self._model, "labels_")
        return self


class KDE:
    """Kernel Density Estimation wrapper using scipy.stats.gaussian_kde.

    Uses the transposed input convention expected by gaussian_kde and exposes simple
    evaluate/resample APIs.
    """

    def __init__(self, bandwidth: Optional[float] = None):
        if not _HAS_SCIPY_STATS:
            raise ImportError("scipy is required for KDE. Install with `pip install scipy`.")
        self.bandwidth = bandwidth
        self.kde = None

    def fit(self, X: np.ndarray) -> "KDE":
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # gaussian_kde expects shape (n_dimensions, n_samples)
        self.kde = gaussian_kde(X.T, bw_method=self.bandwidth)
        return self

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.kde is None:
            raise ValueError("KDE: call fit(...) before evaluate(...)")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.kde(X.T)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        if self.kde is None:
            raise ValueError("KDE: call fit(...) before sample(...)")
        return self.kde.resample(n_samples).T


__all__ = ["DBSCAN", "OPTICS", "KDE"]