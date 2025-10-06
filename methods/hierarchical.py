# hierarchical.py
"""Hierarchical clustering algorithms: agglomerative and a simple divisive approach.

This module keeps the training data in-memory so that predict(...) can compute centroids
consistently from the fitted dataset (important for web interactive workflows).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class AgglomerativeClustering:
    """Agglomerative clustering wrapper around scipy hierarchical linkage.

    Stores training data and computed flat-cluster labels; predict assigns new samples to
    the nearest fitted-cluster centroid.
    """

    def __init__(self, n_clusters: int = 3, method: str = "ward"):
        self.n_clusters = int(n_clusters)
        self.method = method
        self.clusters: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "AgglomerativeClustering":
        X = np.asarray(X)
        self._X_train = X
        # linkage expects a condensed distance matrix when provided with a 1D array
        Z = linkage(X, method=self.method)
        self.clusters = fcluster(Z, t=self.n_clusters, criterion="maxclust")
        logger.info("Agglomerative produced %d clusters", len(np.unique(self.clusters)))
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if self.clusters is None or self._X_train is None:
            raise ValueError("AgglomerativeClustering: call fit(...) before predict(...)")
        X_new = np.asarray(X_new)
        # compute centroids from training set
        centroids = np.array([self._X_train[self.clusters == i].mean(axis=0) for i in range(1, self.n_clusters + 1)])
        distances = np.linalg.norm(X_new[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1) + 1


class DivisiveClustering:
    """A simple top-down divisive clustering approach using k-means to split clusters.

    This implementation maintains unique cluster ids as it splits clusters until the
    requested number of clusters is reached.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.clusters: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DivisiveClustering":
        X = np.asarray(X)
        n_samples = X.shape[0]
        # start with one cluster id = 1
        labels = np.ones(n_samples, dtype=int)
        next_cluster_id = 2

        # split until desired number of clusters
        while labels.max() < self.n_clusters:
            # pick cluster to split: largest cluster
            counts = np.bincount(labels)
            largest_cluster = np.argmax(counts[1:]) + 1  # offset because labels start at 1
            indices = np.where(labels == largest_cluster)[0]
            if len(indices) <= 1:
                break
            kmeans = KMeans(n_clusters=2, random_state=self.random_state)
            sublabels = kmeans.fit_predict(X[indices])
            # assign new ids: keep one piece as the old id, the other gets a new id
            labels[indices[sublabels == 1]] = next_cluster_id
            next_cluster_id += 1
            # safety: if we somehow exceed desired clusters, stop
            if labels.max() >= self.n_clusters:
                break

        self.clusters = labels
        self._X_train = X
        logger.info("Divisive produced %d clusters", len(np.unique(labels)))
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if self.clusters is None or self._X_train is None:
            raise ValueError("DivisiveClustering: call fit(...) before predict(...)")
        X_new = np.asarray(X_new)
        centroids = np.array([self._X_train[self.clusters == i].mean(axis=0) for i in range(1, self.clusters.max() + 1)])
        distances = np.linalg.norm(X_new[:, None, :] - centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1) + 1

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).clusters


__all__ = ["AgglomerativeClustering", "DivisiveClustering"]
