# preprocessor/etl.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class ETL:
    def __init__(self, scaling="standard"):
        self.scaling = scaling
        self.scaler = None
        self.encoders = {}

    def clean_missing(self, df: pd.DataFrame, strategy="drop", fill_value=None):
        if strategy == "drop":
            return df.dropna()
        elif strategy == "mean":
            return df.fillna(df.mean(numeric_only=True))
        elif strategy == "median":
            return df.fillna(df.median(numeric_only=True))
        elif strategy == "mode":
            return df.fillna(df.mode().iloc[0])
        elif strategy == "constant":
            return df.fillna(fill_value)
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

    def encode_categoricals(self, df: pd.DataFrame):
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        return df

    def scale_features(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.scaling == "standard":
            self.scaler = StandardScaler()
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
        else:
            return df  # no scaling
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def transform(self, df: pd.DataFrame, missing="drop"):
        df = self.clean_missing(df, strategy=missing)
        df = self.encode_categoricals(df)
        df = self.scale_features(df)
        return df
