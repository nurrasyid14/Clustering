import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from preprocessor.etl import ETL
from preprocessor.eda import EDA
from methods.visualizer import Visualizer
from methods.evaluator import Evaluator
from methods.centroids import KMeansClustering, FuzzyCMeansClustering, KModesClustering
from methods.densities import DBSCAN, KDE
from methods.distributions_c import GMixtures
from methods.hierarchical import AgglomerativeClustering, DivisiveClustering

st.set_page_config(page_title="Clustering Specialist", layout="wide")

st.title("Clustering Specialist")
st.write("Muhamad Nur Rasyid // 3324600018 // 2 D4 SDT-A")

# Helper to safely show a plotly figure or a list of figures
def _plotly_show(fig_or_list):
    if fig_or_list is None:
        return
    if isinstance(fig_or_list, (list, tuple)):
        for fig in fig_or_list:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig_or_list, use_container_width=True)


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Data Dashboard", "Clustering", "Evaluation", "Perbandingan Metode"])

# --- Tab 1: Data Dashboard ---
with tab1:
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {uploaded_file.name} ({df.shape[0]} baris, {df.shape[1]} kolom)")

        st.sidebar.subheader("Preprocessing Options")
        missing_strategy = st.sidebar.selectbox("Missing Value Handling", ["drop", "mean", "median", "mode", "constant"])
        scaling = st.sidebar.selectbox("Scaling", ["standard", "minmax", "none"])

        etl = ETL(scaling=scaling)
        clean_df = etl.transform(df, missing=missing_strategy)

        st.session_state["df"] = df
        st.session_state["clean_df"] = clean_df

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
        _plotly_show(corr_fig)

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

        # Centroid-based
        if basis == "Centroid-based":
            st.subheader("Metode Berbasis Centroid")
            st.caption("Membagi data menjadi beberapa kelompok dengan mencari titik pusat (centroid).")
            method = st.selectbox("Metode:", ["KMeans", "Fuzzy C-Means", "KModes"])
            n_clusters = st.slider("Jumlah klaster", 2, 10, 3)

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

        # Density-based
        elif basis == "Density-based":
            st.subheader("Metode Berbasis Kepadatan")
            st.caption("Mengelompokkan titik data yang berdekatan dalam area padat; titik yang jauh dianggap noise.")
            method = st.selectbox("Metode:", ["DBSCAN", "KDE"])
            if method == "DBSCAN":
                eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                min_samples = st.slider("Minimum Samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                labels = model.labels_
            elif method == "KDE":
                bandwidth = st.slider("Bandwidth", 0.1, 1.0, 0.2)
                model = KDE(bandwidth=bandwidth).fit(X)
                labels = None

        # Distribution-based
        elif basis == "Distribution-based":
            st.subheader("Metode Berbasis Distribusi")
            st.caption("Memodelkan distribusi data (Gaussian Mixture).")
            n_components = st.slider("Jumlah komponen", 2, 10, 3)
            model = GMixtures(n_components=n_components).fit(X)
            labels = model.labels_

        # Hierarchical
        elif basis == "Hierarchical":
            st.subheader("Metode Hierarkis")
            st.caption("Agglomerative (bottom-up) atau Divisive (top-down).")
            method = st.selectbox("Metode:", ["Agglomerative", "Divisive"])
            n_clusters = st.slider("Jumlah klaster", 2, 10, 3)
            if method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)
            else:
                model = DivisiveClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)

        # Results + Visualization (safe plotting)
        if labels is not None:
            st.write(f"**Algoritma yang Dipilih:** {method}")
            st.write("Cluster Labels:", labels)

            viz = Visualizer()
            _plotly_show(viz.scatter(X, labels))

            # violin returns either list of fig or a single fig; use safe helper
            violin_figs = viz.violin(X, labels)
            _plotly_show(violin_figs)

        st.session_state["labels"] = labels if labels is not None else None
    else:
        st.warning("⚠️ Silakan upload dan pra-proses dataset di tab Data Dashboard.")


# --- Tab 3: Evaluation ---
with tab3:
    if "clean_df" in st.session_state and "labels" in st.session_state and st.session_state["labels"] is not None:
        st.subheader("Evaluasi Kualitas Klastering")
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values
        labels = st.session_state["labels"]

        eval_model = Evaluator(X, labels)
        metrics = eval_model.evaluate_all()

        st.write("### Tabel Metrik Evaluasi")
        st.dataframe(pd.DataFrame(metrics, index=[0]).T)

        viz = Visualizer()
        _plotly_show(viz.metrics_bar(metrics))
        _plotly_show(viz.metrics_radar(metrics))

        # Footnote / catatan (expander)
        with st.expander("Penjelasan Metrik Evaluasi"):
            st.markdown("""
            **Silhouette Score**  
            Mengukur seberapa mirip suatu titik dengan klasternya sendiri dibandingkan dengan klaster lain.  
            - Range: **-1 sampai 1**. Semakin tinggi (mendekati 1) semakin baik.

            **Davies-Bouldin Index (DBI)**  
            Mengukur seberapa mirip antar klaster (rasio intra-cluster / inter-cluster).  
            - Nilai lebih kecil lebih baik.

            **Calinski-Harabasz Index (CHI)**  
            Mengukur rasio variasi antar-klaster terhadap variasi intra-klaster.  
            - Nilai lebih besar lebih baik.
            """)
    else:
        st.warning("⚠️ Silakan selesaikan proses clustering terlebih dahulu.")


# --- Tab 4: Perbandingan Metode ---
with tab4:
    st.subheader("Perbandingan Metode Clustering")
    st.caption("Perbandingan metrik antar metode.")

    if "clean_df" in st.session_state:
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values

        # Let user choose n_clusters for comparators that need it
        n_clusters = st.slider("Default jumlah klaster untuk perbandingan (dipakai oleh metode berbasis centroid/hierarki)", 2, 10, 3)

        # Prepare methods
        methods = {
            "KMeans": lambda: KMeansClustering(n_clusters=n_clusters),
            "Fuzzy C-Means": lambda: FuzzyCMeansClustering(n_clusters=n_clusters),
            "KModes": lambda: KModesClustering(n_clusters=n_clusters),
            "GaussianMixture": lambda: GMixtures(n_components=n_clusters),
            "Agglomerative": lambda: AgglomerativeClustering(n_clusters=n_clusters),
            "Divisive": lambda: DivisiveClustering(n_clusters=n_clusters),
            "DBSCAN": lambda: DBSCAN(eps=0.5, min_samples=5)
        }

        if st.button("Jalankan Perbandingan Metode"):
            results = []
            for name, factory in methods.items():
                model = None
                labels = None
                try:
                    model = factory()
                    # Fit depending on API
                    fit_ret = model.fit(X)
                    # Prefer standardized label extraction
                    if hasattr(model, "predict"):
                        labels = model.predict(X)
                    elif hasattr(model, "labels_"):
                        labels = getattr(model, "labels_")
                    elif isinstance(fit_ret, (list, np.ndarray)):
                        labels = np.asarray(fit_ret)
                    else:
                        # try fit_predict fallback
                        if hasattr(model, "fit_predict"):
                            labels = model.fit_predict(X)
                        elif hasattr(model, "fit") and hasattr(model, "clusters"):
                            labels = getattr(model, "clusters")
                except Exception as e:
                    st.warning(f"Gagal menjalankan {name}: {e}")
                    labels = None

                # Normalize labels to numpy or mark as invalid
                if labels is None:
                    silhouette = davies = calinski = np.nan
                else:
                    try:
                        labels = np.asarray(labels)
                        # Remove noise-only or single-cluster cases
                        unique_labels = set(labels.flatten())
                        if len(unique_labels) <= 1 or (len(unique_labels) == 1 and -1 in unique_labels):
                            silhouette = davies = calinski = np.nan
                        else:
                            # For DBSCAN where noise label is -1, keep computation (silhouette ignores -1 automatically)
                            silhouette = silhouette_score(X, labels)
                            davies = davies_bouldin_score(X, labels)
                            calinski = calinski_harabasz_score(X, labels)
                    except Exception as e:
                        st.warning(f"Gagal menghitung metrik untuk {name}: {e}")
                        silhouette = davies = calinski = np.nan

                results.append({
                    "Metode": name,
                    "Silhouette": silhouette,
                    "Davies-Bouldin": davies,
                    "Calinski-Harabasz": calinski
                })

            df_results = pd.DataFrame(results).set_index("Metode")
            st.write("### Hasil Perbandingan (nilai NaN berarti metrik tidak dapat dihitung)")
            st.dataframe(df_results.style.format("{:.4f}"))

            # Highlight bests
            best_silhouette = df_results["Silhouette"].idxmax() if df_results["Silhouette"].notna().any() else None
            best_davies = df_results["Davies-Bouldin"].idxmin() if df_results["Davies-Bouldin"].notna().any() else None
            best_calinski = df_results["Calinski-Harabasz"].idxmax() if df_results["Calinski-Harabasz"].notna().any() else None

            st.markdown("### Rangkuman Otomatis")
            if best_silhouette:
                st.write(f"- Silhouette terbaik: **{best_silhouette}**")
            if best_davies:
                st.write(f"- Davies-Bouldin terbaik (terkecil): **{best_davies}**")
            if best_calinski:
                st.write(f"- Calinski-Harabasz terbaik: **{best_calinski}**")

            # footnote
            with st.expander("Catatan: Penjelasan metrik evaluasi"):
                st.markdown("""
                **Silhouette Score**  
                Mengukur seberapa mirip titik dengan klasternya sendiri dibandingkan dengan klaster lain.
                - Range: -1 sampai 1. Semakin tinggi semakin baik.

                **Davies-Bouldin Index (DBI)**  
                Mengukur kemiripan antar klaster; nilai lebih kecil lebih baik.

                **Calinski-Harabasz Index (CHI)**  
                Mengukur rasio variasi antar-klaster terhadap variasi intra-klaster; nilai lebih besar lebih baik.

                Catatan: tidak semua metode menghasilkan label valid pada setiap pengaturan (mis. DBSCAN bisa memberi banyak noise).
                """)
    else:
        st.warning("⚠️ Silakan unggah dan pra-proses dataset terlebih dahulu di tab Data Dashboard.")
