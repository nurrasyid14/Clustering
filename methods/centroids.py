# centroids.py
"""
Centroid-based clustering wrappers: KMeans, Fuzzy C-Means (skfuzzy), KModes (optional).
Provides a small, consistent API around popular clustering algorithms with type hints,
robust input handling, and small performance improvements.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

try:
    from skfuzzy.cluster import cmeans
    from skfuzzy import cmeans_predict
    _HAS_SKFUZZY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SKFUZZY = False

try:
    from kmodes.kmodes import KModes
    _HAS_KMODES = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_KMODES = False


logger = logging.getLogger(__name__)


class KMeansClustering:
    """Light wrapper around sklearn.cluster.KMeans with a consistent interface.

    Methods
    -------
    fit(X) -> self
    predict(X) -> np.ndarray
    get_centroids() -> np.ndarray
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42, **kwargs):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, **kwargs)
        self._is_fitted = False
        logger.debug("KMeans initialized: %s", {"n_clusters": self.n_clusters})

    def fit(self, X: np.ndarray) -> "KMeansClustering":
        X = np.asarray(X)
        self.model.fit(X)
        self._is_fitted = True
        self._X_train = X
        logger.info("KMeans fitted on %d samples", X.shape[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if not self._is_fitted:
            raise ValueError("KMeansClustering: call fit(...) before predict(...)")
        return self.model.predict(X)

    def get_centroids(self) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("KMeansClustering: call fit(...) before get_centroids(...)")
        return self.model.cluster_centers_


class FuzzyCMeansClustering:
    """Wrapper around skfuzzy c-means. Optional dependency."""

    def __init__(
        self,
        n_clusters: int = 3,
        m: float = 2.0,
        error: float = 1e-5,
        maxiter: int = 1000,
        seed: Optional[int] = 42,
    ):
        if not _HAS_SKFUZZY:
            raise ImportError(
                "scikit-fuzzy is required for FuzzyCMeansClustering. "
                "Install with `pip install scikit-fuzzy`."
            )

        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.error = float(error)
        self.maxiter = int(maxiter)
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "FuzzyCMeansClustering":
        """Fit fuzzy c-means on data."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # scikit-fuzzy returns between 5 and 7 values depending on version
        result = cmeans(
            X.T,
            c=self.n_clusters,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter,
            seed=self.seed,
        )

        # Unpack flexibly
        cntr, u = result[0], result[1]

        self.centroids = np.asarray(cntr)
        self.u = np.asarray(u)
        self._is_fitted = True
        return self


    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster memberships for new data."""
        if not self._is_fitted:
            raise ValueError("FuzzyCMeansClustering: call fit(...) before predict(...)")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        result = cmeans_predict(
            X.T,
            self.centroids,
            m=self.m,
            error=self.error,
            maxiter=self.maxiter,
        )

        # Handle variable length again
        u_pred = result[0]

        labels = np.argmax(u_pred, axis=0)
        return labels




class KModesClustering:
    """Light wrapper around kmodes.KModes. Optional dependency.

    Uses the KModes implementation when available.
    """

    def __init__(self, n_clusters: int = 3, init: str = "Huang", n_init: int = 5, random_state: Optional[int] = 42, verbose: int = 0):
        if not _HAS_KMODES:
            raise ImportError("kmodes is required for KModesClustering. Install with `pip install kmodes`.")
        self.n_clusters = int(n_clusters)
        self.model = KModes(n_clusters=self.n_clusters, init=init, n_init=n_init, random_state=random_state, verbose=verbose)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "KModesClustering":
        self.model.fit(X)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("KModesClustering: call fit(...) before predict(...)")
        return self.model.predict(X)

    def get_centroids(self) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("KModesClustering: not fitted")
        return np.asarray(self.model.cluster_centroids_)


__all__ = ["KMeansClustering", "FuzzyCMeansClustering", "KModesClustering"]