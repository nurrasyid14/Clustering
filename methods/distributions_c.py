# distributions_c.py
"""Distribution-based clustering wrappers (Gaussian Mixture Models).

Includes additional convenience methods and safe handling when silhouette score cannot be computed.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class GMixtures:
    def __init__(self, n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.random_state = int(random_state)
        self.model = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type, random_state=self.random_state)
        self.labels_: Optional[np.ndarray] = None
        self.silhouette_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "GMixtures":
        X = np.asarray(X)
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        if len(set(self.labels_)) > 1:
            try:
                self.silhouette_ = float(silhouette_score(X, self.labels_))
            except Exception:
                self.silhouette_ = None
        else:
            self.silhouette_ = None
        logger.info("GMixtures fitted: %d components", self.n_components)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
            "silhouette_score": self.silhouette_,
        }

    def get_silhouette_score(self) -> Optional[float]:
        return self.silhouette_

    def get_cluster_centers(self) -> np.ndarray:
        return getattr(self.model, "means_", None)

    def get_covariances(self) -> np.ndarray:
        return getattr(self.model, "covariances_", None)

    def get_weights(self) -> np.ndarray:
        return getattr(self.model, "weights_", None)

    def get_model(self) -> GaussianMixture:
        return self.model


__all__ = ["GMixtures"]
