# preprocessor/eda.py
import pandas as pd
import numpy as np

class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summary(self):
        return {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "missing": self.df.isnull().sum().to_dict(),
            "describe": self.df.describe(include="all").to_dict()
        }

    def correlations(self, method="pearson"):
        return self.df.corr(method=method)

    def value_counts(self, column: str, normalize=False):
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in dataset")
        return self.df[column].value_counts(normalize=normalize).to_dict()

    def detect_outliers(self, z_thresh=3):
        numeric = self.df.select_dtypes(include=[np.number])
        z_scores = (numeric - numeric.mean()) / numeric.std()
        return (np.abs(z_scores) > z_thresh).sum().to_dict()
