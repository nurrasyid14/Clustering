import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class Evaluator:
    def __init__(self, X, labels):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X = X
        self.labels = labels

    def silhouette(self):
        """Silhouette Score (higher is better, max=1)."""
        try:
            return silhouette_score(self.X, self.labels)
        except Exception:
            return np.nan

    def calinski_harabasz(self):
        """Calinski-Harabasz Index (higher is better)."""
        try:
            return calinski_harabasz_score(self.X, self.labels)
        except Exception:
            return np.nan

    def davies_bouldin(self):
        """Davies-Bouldin Index (lower is better)."""
        try:
            return davies_bouldin_score(self.X, self.labels)
        except Exception:
            return np.nan

    def evaluate_all(self):
        """Return all metrics as a dictionary."""
        return {
            "Silhouette Score": self.silhouette(),
            "Calinski-Harabasz": self.calinski_harabasz(),
            "Davies-Bouldin": self.davies_bouldin()
        }


__all__ = ["Evaluator"]
