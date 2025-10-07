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
    uploaded_files = st.file_uploader("Upload satu atau lebih file CSV/XLSX", type=["csv", "xlsx"], accept_multiple_files=True)
    if uploaded_files:
        if "datasets" not in st.session_state:
            st.session_state["datasets"] = {}
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file) 
            st.success(f"✅ Loaded {uploaded_file.name} ({df.shape[0]} baris, {df.shape[1]} kolom)")
            # Preprocessing 
            etl = ETL(scaling="standard") 
            clean_df = etl.transform(df, missing="mean") 
            st.session_state["datasets"][uploaded_file.name] = clean_df 
            # EDA summary 
            eda = EDA(df) 
            summary = eda.summary() 
            st.markdown(f"### Dataset: `{uploaded_file.name}`") 
            st.write("**Shape:**", summary["shape"]) 
            st.write("**Missing Values:**", summary["missing"]) 
            st.dataframe(pd.DataFrame(summary["describe"])) 
            st.write("**Correlation Matrix:**") 
            viz = Visualizer() 
            corr_fig = viz.heatmap(eda.correlations()) 
            _plotly_show(corr_fig) 

        st.info("Semua dataset telah diproses dan disimpan di session_state.")
    else:
        st.info("Upload dataset terlebih dahulu untuk melihat dashboard.")

# --- Tab 2: Clustering ---
with tab2:
    if "datasets" in st.session_state and st.session_state["datasets"]:
        dataset_names = list(st.session_state["datasets"].keys())
        selected_dataset_name = st.selectbox("Pilih dataset untuk clustering", dataset_names)
        clean_df = st.session_state["datasets"][selected_dataset_name]
        X = clean_df.select_dtypes(include="number").values

        st.subheader("Clustering Pipeline")
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

        # Results + Visualization
        if labels is not None:
            st.write(f"**Algoritma yang Dipilih:** {method}")
            st.write("Cluster Labels:", labels)

            viz = Visualizer()
            _plotly_show(viz.scatter(X, labels))
            violin_figs = viz.violin(X, labels)
            _plotly_show(violin_figs)

            # Save labels per dataset
            if "cluster_labels" not in st.session_state:
                st.session_state["cluster_labels"] = {}
            st.session_state["cluster_labels"][selected_dataset_name] = labels
    else:
        st.warning("⚠️ Silakan upload dan pra-proses dataset di tab Data Dashboard.")

# --- Tab 3: Evaluation ---
with tab3:
    if "datasets" in st.session_state and st.session_state["datasets"]:
        dataset_names = list(st.session_state["datasets"].keys())
        selected_dataset_name = st.selectbox("Pilih dataset untuk evaluasi", dataset_names)
        clean_df = st.session_state["datasets"][selected_dataset_name]
        X = clean_df.select_dtypes(include="number").values
        labels = st.session_state.get("cluster_labels", {}).get(selected_dataset_name)

        if labels is not None:
            st.subheader("Evaluasi Kualitas Klastering")
            eval_model = Evaluator(X, labels)
            metrics = eval_model.evaluate_all()

            st.write("### Tabel Metrik Evaluasi")
            st.dataframe(pd.DataFrame(metrics, index=[0]).T)

            viz = Visualizer()
            _plotly_show(viz.metrics_bar(metrics))
            _plotly_show(viz.metrics_radar(metrics))

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
            st.warning("⚠️ Dataset ini belum di-cluster. Silakan lakukan clustering di Tab 2.")
    else:
        st.warning("⚠️ Silakan upload dan pra-proses dataset di tab Data Dashboard.")

# --- Tab 4: Perbandingan Metode dan Dataset ---
with tab4: 
    st.subheader("Perbandingan Dataset dan Metode Clustering") 
    st.caption("Bandingkan performa clustering antar dataset dan antar metode.")
    
    if "datasets" in st.session_state and st.session_state["datasets"]:
        dataset_names = list(st.session_state["datasets"].keys())
        selected_datasets = st.multiselect(
            "Pilih dataset untuk dibandingkan", dataset_names, default=dataset_names
        )

        if st.button("Bandingkan Dataset & Metode"): 
            # Definisikan metode clustering
            methods = [
                ("KMeans", lambda X: KMeansClustering(n_clusters=3).fit(X).predict(X)),
                ("Fuzzy C-Means", lambda X: FuzzyCMeansClustering(n_clusters=3, m=2).fit(X).predict(X)),
                ("KModes", lambda X: KModesClustering(n_clusters=3).fit(X).predict(X)),
                ("DBSCAN", lambda X: DBSCAN(eps=0.5, min_samples=5).fit(X).labels_),
                ("GMM", lambda X: GMixtures(n_components=3).fit(X).labels_),
                ("Agglomerative", lambda X: AgglomerativeClustering(n_clusters=3).fit_predict(X)),
                ("Divisive", lambda X: DivisiveClustering(n_clusters=3).fit_predict(X))
            ]

            results = []

            for dataset_name in selected_datasets:
                clean_df = st.session_state["datasets"][dataset_name]
                X = clean_df.select_dtypes(include="number").values

                for method_name, clustering_func in methods:
                    try:
                        labels = clustering_func(X)
                        silhouette = silhouette_score(X, labels)
                        davies = davies_bouldin_score(X, labels)
                        calinski = calinski_harabasz_score(X, labels)
                    except Exception:
                        silhouette = davies = calinski = np.nan

                    results.append({
                        "Dataset": dataset_name,
                        "Method": method_name,
                        "Silhouette": silhouette,
                        "Davies-Bouldin": davies,
                        "Calinski-Harabasz": calinski
                    })

            # Buat DataFrame hasil
            df_results = pd.DataFrame(results)
            st.write("### Hasil Perbandingan Metode Clustering per Dataset")
            st.dataframe(
                df_results.style.format({
                    "Silhouette": "{:.4f}",
                    "Davies-Bouldin": "{:.4f}",
                    "Calinski-Harabasz": "{:.4f}"
                })
            )

            # Tampilkan metode terbaik per dataset
            for dataset_name in selected_datasets:
                subset = df_results[df_results["Dataset"] == dataset_name]
                best_sil = subset.loc[subset["Silhouette"].idxmax(), "Method"]
                best_dav = subset.loc[subset["Davies-Bouldin"].idxmin(), "Method"]
                best_cal = subset.loc[subset["Calinski-Harabasz"].idxmax(), "Method"]

                st.markdown(f"**Dataset {dataset_name} - Ringkasan Metode Terbaik:**")
                st.write(f"• Silhouette terbaik: **{best_sil}**")
                st.write(f"• Davies-Bouldin terbaik (terkecil): **{best_dav}**")
                st.write(f"• Calinski-Harabasz terbaik: **{best_cal}**")

            with st.expander("📘 Catatan Metrik"):
                st.markdown(""" 
                **Silhouette Score:** semakin tinggi semakin baik (maksimum = 1).  
                **Davies-Bouldin Index:** semakin rendah semakin baik (minimum = 0).  
                **Calinski-Harabasz Index:** semakin tinggi semakin baik.  
                """)
    else:
        st.warning("⚠️ Silakan unggah dan pra-proses dataset terlebih dahulu di tab Data Dashboard.")
