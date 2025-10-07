# app.py
import streamlit as st
import pandas as pd

from preprocessor.etl import ETL
from preprocessor.eda import EDA
from methods.visualizer import Visualizer
from methods.evaluator import Evaluator
from methods.centroids import KMeansClustering, FuzzyCMeansClustering, KModesClustering
from methods.densities import DBSCAN, KDE
from methods.distributions_c import GMixtures
from methods.hierarchical import AgglomerativeClustering, DivisiveClustering

st.set_page_config(page_title="Clustering Specialist", layout="wide")

# --- App Header ---
st.title("🤖 Clustering Specialist")
st.write("Upload your dataset, preprocess it, explore clustering methods, and evaluate performance.")

# --- Tabs ---
import openai

# Add this new tab:
tab1, tab2, tab3, tab4 = st.tabs(["Data Dashboard", "Clustering", "Evaluation", "Perbandingan Metode"])

# --- Tab 1: Data Dashboard ---
with tab1:
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")

        # Sidebar preprocessing options
        st.sidebar.subheader("Preprocessing Options")
        missing_strategy = st.sidebar.selectbox("Missing Value Handling", ["drop", "mean", "median", "mode", "constant"])
        scaling = st.sidebar.selectbox("Scaling", ["standard", "minmax", "none"])

        etl = ETL(scaling=scaling)
        clean_df = etl.transform(df, missing=missing_strategy)

        st.session_state["df"] = df
        st.session_state["clean_df"] = clean_df

        # EDA Dashboard
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
        if isinstance(corr_fig, list):
            for f in corr_fig:
                st.plotly_chart(f, use_container_width=True)
        else:
            st.plotly_chart(corr_fig, use_container_width=True)

    else:
        st.info("Upload a dataset to see the dashboard.")


# --- Tab 2: Clustering ---
with tab2:
    
    if "clean_df" in st.session_state:
        st.subheader("Clustering Pipeline")
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values
        labels, method = None, None

        # Basis first
        basis = st.radio("Select clustering basis:", ["Centroid-based", "Density-based", "Distribution-based", "Hierarchical"])

        # Centroid
        if basis == "Centroid-based":
            st.subheader("Metode Berbasis Centroid")
            st.caption("Metode ini membagi data ke dalam beberapa kelompok dengan mencari pusat (centroid) "
               "yang meminimalkan jarak total antara titik data dan centroid-nya.")
            method = st.selectbox("Choose method:", ["KMeans", "Fuzzy C-Means", "KModes"])
            n_clusters = st.slider("Number of clusters", 2, 10, 3)

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


        # Density
        elif basis == "Density-based":
            st.subheader("Metode Berbasis Kepadatan")
            st.caption("Metode pengelompokan titik-titik data yang berdekatan dalam area yang padat, "
               "sementara titik yang jauh dianggap sebagai noise. Cocok untuk data dengan bentuk tidak beraturan.")
    
            method = st.selectbox("Choose method:", ["DBSCAN", "KDE"])
            if method == "DBSCAN":
                eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                min_samples = st.slider("Minimum Samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                labels = model.labels_
            elif method == "KDE":
                bandwidth = st.slider("Bandwidth", 0.1, 1.0, 0.2)
                model = KDE(bandwidth=bandwidth).fit(X)
                labels = None

        # Distribution
        elif basis == "Distribution-based":
            method = "Gaussian Mixture"
            n_components = st.slider("Number of Components", 2, 10, 3)
            model = GMixtures(n_components=n_components).fit(X)
            labels = model.labels_

        # Hierarchical
        elif basis == "Hierarchical":
            st.subheader("Metode Hierarkis")
            st.caption("Metode ini membentuk hierarki klaster secara bertahap: agglomerative (dari bawah ke atas) "
               "atau divisive (dari atas ke bawah). Hasilnya dapat divisualisasikan dengan dendrogram.")
            method = st.selectbox("Choose method:", ["Agglomerative", "Divisive"])
            n_clusters = st.slider("Number of clusters", 2, 10, 3)
            if method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)
            else:
                model = DivisiveClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)



        # Results + Visualization
        if labels is not None:
            st.write(f"**Selected Algorithm:** {method}")
            st.write("Cluster Labels:", labels)

            viz = Visualizer()
            scatter_fig = viz.scatter(X, labels)
            st.plotly_chart(scatter_fig, use_container_width=True)

            violin_figs = viz.violin(X, labels)
            if isinstance(violin_figs, list):
                for f in violin_figs:
                    st.plotly_chart(f, use_container_width=True)
            else:
                st.plotly_chart(violin_figs, use_container_width=True)

        st.session_state["labels"] = labels if labels is not None else None
    else:
        st.warning("⚠️ Please upload and preprocess a dataset in the Data Dashboard first.")


# --- Tab 3: Evaluation ---
with tab3:
    if "clean_df" in st.session_state and "labels" in st.session_state and st.session_state["labels"] is not None:
        st.subheader("Evaluation Metrics")
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values
        labels = st.session_state["labels"]

        eval_model = Evaluator(X, labels)
        metrics = eval_model.evaluate_all()

        st.write("### Metrics Table")
        st.dataframe(pd.DataFrame(metrics, index=[0]).T)

        viz = Visualizer()
        for fig in [viz.metrics_bar(metrics), viz.metrics_radar(metrics)]:
            if isinstance(fig, list):
                for f in fig:
                    st.plotly_chart(f, use_container_width=True)
            else:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please complete clustering before evaluation.")

# --- Tab 4: Perbandingan Metode ---
with tab4:
    st.subheader("Perbandingan Metode Klastering")
    st.caption("Analisis otomatis untuk menentukan metode klastering terbaik berdasarkan karakteristik data.")

    if "clean_df" in st.session_state:
        clean_df = st.session_state["clean_df"]
        X = clean_df.select_dtypes(include="number").values

        if st.button("Analisis Otomatis dengan OpenAI"):
            with st.spinner("Menganalisis dataset menggunakan OpenAI..."):
                # Summarize dataset briefly for context
                summary = (
                    f"Dataset dengan {X.shape[0]} baris dan {X.shape[1]} fitur. "
                    f"Rata-rata: {np.mean(X):.2f}, standar deviasi: {np.std(X):.2f}."
                )

                # Compose a concise prompt
                prompt = f"""
                Berdasarkan deskripsi dataset berikut:
                {summary}

                Jelaskan secara singkat metode klastering mana yang paling sesuai:
                - KMeans
                - Fuzzy C-Means
                - KModes
                - DBSCAN
                - OPTICS
                - Agglomerative
                - Divisive

                Gunakan bahasa Indonesia yang sederhana.
                """

                # OpenAI API call
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                explanation = response.choices[0].message.content
                st.markdown("### 🔍 Hasil Analisis OpenAI")
                st.markdown(explanation)
    else:
        st.warning("⚠️ Silakan unggah dan pra-proses dataset terlebih dahulu di tab Data Dashboard.")

