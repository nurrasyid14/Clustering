```python
import streamlit as st
import pandas as pd
import numpy as np

from preprocessor.etl import ETL
from preprocessor.eda import EDA
from methods.visualizer import Visualizer
from methods.evaluator import Evaluator
from methods.centroids import KMeansClustering, FuzzyCMeansClustering, KModesClustering
from methods.densities import DBSCAN, KDE
from methods.distributions_c import GMixtures
from methods.hierarchical import AgglomerativeClustering, DivisiveClustering


# --- Page Config ---
st.set_page_config(page_title="Clustering Specialist", layout="wide")

# --- App Header ---
st.title("🤖 Clustering Specialist")
st.write("Upload dataset, lakukan preprocessing, eksplor berbagai metode clustering, dan evaluasi performanya.")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Data Dashboard", "Clustering", "Evaluation", "Perbandingan Metode"])

# --- Tab 1: Data Dashboard ---
with tab1:
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {uploaded_file.name} ({df.shape[0]} baris, {df.shape[1]} kolom)")

        # Sidebar preprocessing
        st.sidebar.subheader("Preprocessing Options")
        missing_strategy = st.sidebar.selectbox("Missing Value Handling", ["drop", "mean", "median", "mode", "constant"])
        scaling = st.sidebar.selectbox("Scaling", ["standard", "minmax", "none"])

        etl = ETL(scaling=scaling)
        clean_df = etl.transform(df, missing=missing_strategy)

        st.session_state["df"] = df
        st.session_state["clean_df"] = clean_df

        # EDA
        eda = EDA(df)
        summary = eda.summary()
        st.write("**Dataset Shape:**", summary["shape"])
        st.write("**Column Types:**", summary["dtypes"])
        st.write("**Missing Values:**", summary["missing"])
        st.write("**Statistics:**")
        st.dataframe(pd.DataFrame(summary["describe"]))
        st.write("**Correlation Matrix:**")
        st.dataframe(eda.correlations())

        viz = Visualizer()
        corr_fig = viz.heatmap(eda.correlations())
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("Upload dataset terlebih dahulu untuk melihat dashboard.")


# --- Tab 2: Clustering ---
with tab2:
    if "clean_df" in st.session_state:
        st.subheader("Clustering Pipeline")
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values
        labels, method = None, None

        basis = st.radio("Pilih jenis metode clustering:", ["Centroid-based", "Density-based", "Distribution-based", "Hierarchical"])

        # --- Centroid-based ---
        if basis == "Centroid-based":
            st.subheader("Metode Berbasis Centroid")
            st.caption("Membagi data menjadi beberapa kelompok dengan mencari titik pusat (centroid) "
                       "yang meminimalkan jarak antara data dan centroid-nya.")
            method = st.selectbox("Metode:", ["KMeans", "Fuzzy C-Means", "KModes"])
            n_clusters = st.slider("Jumlah cluster", 2, 10, 3)

            if method == "KMeans":
                model = KMeansClustering(n_clusters=n_clusters)
                model.fit(X)
                labels = model.predict(X)
            elif method == "Fuzzy C-Means":
                m = st.slider("Fuzziness (m)", 1, 5, 2)
                model = FuzzyCMeansClustering(n_clusters=n_clusters, m=m)
                model.fit(X)
                labels = model.predict(X)
            elif method == "KModes":
                model = KModesClustering(n_clusters=n_clusters)
                model.fit(X)
                labels = model.predict(X)

        # --- Density-based ---
        elif basis == "Density-based":
            st.subheader("Metode Berbasis Kepadatan")
            st.caption("Mengelompokkan titik data yang berdekatan dalam area padat, "
                       "sementara data di luar area tersebut dianggap noise.")
            method = st.selectbox("Metode:", ["DBSCAN", "KDE"])

            if method == "DBSCAN":
                eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                min_samples = st.slider("Minimum Samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                labels = model.labels_
            elif method == "KDE":
                bandwidth = st.slider("Bandwidth", 0.1, 1.0, 0.2)
                model = KDE(bandwidth=bandwidth).fit(X)

        # --- Distribution-based ---
        elif basis == "Distribution-based":
            st.subheader("Metode Berbasis Distribusi")
            st.caption("Mencoba memodelkan distribusi data dengan pendekatan probabilistik menggunakan Gaussian Mixture Models.")
            method = "Gaussian Mixture"
            n_components = st.slider("Jumlah komponen", 2, 10, 3)
            model = GMixtures(n_components=n_components).fit(X)
            labels = model.labels_

        # --- Hierarchical ---
        elif basis == "Hierarchical":
            st.subheader("Metode Hierarkis")
            st.caption("Membentuk hierarki cluster: agglomerative (bawah ke atas) "
                       "atau divisive (atas ke bawah). Cocok untuk eksplorasi struktur data.")
            method = st.selectbox("Metode:", ["Agglomerative", "Divisive"])
            n_clusters = st.slider("Jumlah cluster", 2, 10, 3)

            if method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)
            else:
                model = DivisiveClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)

        # --- Visualization ---
        if labels is not None:
            st.write(f"**Algoritma yang Dipilih:** {method}")
            st.write("Cluster Labels:", labels)

            viz = Visualizer()
            st.plotly_chart(viz.scatter(X, labels), use_container_width=True)
            st.plotly_chart(viz.violin(X, labels), use_container_width=True)

        st.session_state["labels"] = labels if labels is not None else None
    else:
        st.warning("⚠️ Silakan upload dan pra-proses dataset di tab Data Dashboard.")


# --- Tab 3: Evaluation ---
with tab3:
    if "clean_df" in st.session_state and "labels" in st.session_state and st.session_state["labels"] is not None:
        st.subheader("Evaluasi Kualitas Clustering")
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values
        labels = st.session_state["labels"]

        eval_model = Evaluator(X, labels)
        metrics = eval_model.evaluate_all()

        st.write("### Tabel Metrik Evaluasi")
        st.dataframe(pd.DataFrame(metrics, index=[0]).T)

        viz = Visualizer()
        for fig in [viz.metrics_bar(metrics), viz.metrics_radar(metrics)]:
            st.plotly_chart(fig, use_container_width=True)

        # --- Footnote / Catatan Penjelasan ---
        with st.expander("📘 Penjelasan Metrik Evaluasi"):
            st.markdown("""
            **Silhouette Score**  
            Mengukur seberapa mirip suatu data dengan clusternya sendiri dibandingkan dengan cluster lain.  
            - Nilai antara **-1 sampai 1**, makin mendekati **1** berarti semakin baik.

            **Davies-Bouldin Index (DBI)**  
            Mengukur tingkat kemiripan antar cluster.  
            - Nilai **lebih kecil lebih baik**, artinya cluster lebih terpisah dengan jelas.

            **Calinski-Harabasz Index (CHI)**  
            Mengukur rasio antara jarak antar cluster dan jarak dalam cluster.  
            - Nilai **lebih besar lebih baik**, menunjukkan cluster padat dan terpisah jauh.
            """)
    else:
        st.warning("⚠️ Silakan selesaikan proses clustering terlebih dahulu.")


# --- Tab 4: Perbandingan Metode ---
with tab4:
    st.subheader("Perbandingan Metode Clustering")
    st.caption("Tab ini menampilkan perbandingan kinerja berbagai metode clustering berdasarkan metrik evaluasi.")

    if "clean_df" in st.session_state:
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values

        st.info("🚧 Fitur analisis otomatis (AI-based) sedang dinonaktifkan sementara. "
                "Namun, kamu tetap bisa membandingkan hasil metrik dari tab *Evaluation*.")
    else:
        st.warning("⚠️ Silakan unggah dataset terlebih dahulu di tab Data Dashboard.")
```
