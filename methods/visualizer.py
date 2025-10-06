import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self):
        pass

    def heatmap(self, corr_matrix):
        """Correlation heatmap using Plotly."""
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="Viridis",
            title="Correlation Heatmap"
        )
        return fig

    def scatter(self, X, labels, method_name="Cluster Scatter", color_map="Viridis"):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            X = PCA(n_components=2).fit_transform(X)

        df_plot = pd.DataFrame(X, columns=["x", "y"])
        df_plot["cluster"] = labels

        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="cluster",
            title=method_name,
            color_continuous_scale=color_map,
            symbol="cluster"
        )

        fig.update_layout(
            legend=dict(
                orientation="h",   # horizontal legend
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        return fig


    def violin(self, X, labels):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(X.shape[1])])

        X["cluster"] = labels

        figs = []
        for col in X.columns[:-1]:
            fig = px.violin(
                X,
                x="cluster",
                y=col,
                color="cluster",
                box=True,
                points="all",
                title=f"Feature Distribution ({col})",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5
                )
            )
            figs.append(fig)
        return figs


    def metrics_bar(self, results: dict):
        """
        Bar chart comparing clustering metrics.
        results: dict like {"Silhouette": 0.65, "Davies-Bouldin": 0.42, ...}
        """
        df = pd.DataFrame(list(results.items()), columns=["Metric", "Score"])
        
        fig = px.bar(
            df,
            x="Metric",
            y="Score",
            color="Metric",
            text_auto=True,
            title="Clustering Metrics"
        )
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Score")
        return fig


    def metrics_radar(self, results: dict):
        """
        Radar chart for clustering metrics.
        results: dict like {"Silhouette": 0.65, "Davies-Bouldin": 0.42, ...}
        """
        categories = list(results.keys())
        values = list(results.values())

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name="Clustering"
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
            showlegend=False,
            title="Clustering Metrics Radar"
        )
        return fig
