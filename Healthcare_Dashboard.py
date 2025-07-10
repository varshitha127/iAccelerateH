import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load your clustered dataset
DATA_PATH = os.path.join('data', 'integrated_clustered_healthcare_data.csv')
METRICS_PATH = os.path.join('data', 'clustering_metrics.csv')
FEATURE_IMPORTANCE_PATH = os.path.join('data', 'rf_feature_importances.csv')
PCA_EXPLAINED_VAR_PATH = os.path.join('data', 'pca_explained_variance.csv')

def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {path}. Please run the backend script first.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data(DATA_PATH)

def load_metrics(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Could not load clustering metrics: {e}")
        return None

metrics_df = load_metrics(METRICS_PATH)

def load_feature_importances(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Could not load feature importances: {e}")
        return None

feature_importances_df = load_feature_importances(FEATURE_IMPORTANCE_PATH)

def load_pca_explained_var(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.warning(f"Could not load PCA explained variance: {e}")
        return None

pca_explained_var_df = load_pca_explained_var(PCA_EXPLAINED_VAR_PATH)

if df is not None and all(col in df.columns for col in ["Cluster_GMM", "BMI", "Recommendation"]):
    # Streamlit App
    st.title("Personalized Healthcare Dashboard")

    # Sidebar Filters
    st.sidebar.title("User Filters")
    selected_cluster = st.sidebar.selectbox("Select Cluster:", df["Cluster_GMM"].unique())

    if metrics_df is not None:
        st.sidebar.markdown("## Clustering Evaluation Metrics")
        for _, row in metrics_df.iterrows():
            st.sidebar.write(f"**{row['Clustering']}**")
            st.sidebar.write(f"Silhouette Score: {row['Silhouette Score']:.3f}")
            st.sidebar.write(f"Davies-Bouldin Index: {row['Davies-Bouldin Index']:.3f}")

    # Advanced EDA: Correlation Heatmap
    st.write("### Correlation Heatmap (Numerical Features)")
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Advanced EDA: Pairplot (sampled for performance)
    st.write("### Pairplot (Sampled Data)")
    sample_df = df.sample(min(200, len(df)), random_state=42) if len(df) > 200 else df
    pairplot_fig = sns.pairplot(sample_df, vars=[col for col in num_cols if col != 'Cluster_GMM'], hue="Cluster_GMM", diag_kind="kde")
    st.pyplot(pairplot_fig)

    # Show Cluster-Based Insights
    st.write("### Cluster Distribution")
    fig = px.histogram(df, x="Cluster_GMM", color="Cluster_GMM", title="Cluster Distribution")
    st.plotly_chart(fig)

    # Show Boxplot for Health Factors
    st.write("### BMI Across Clusters")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df["Cluster_GMM"], y=df["BMI"], ax=ax)
    st.pyplot(fig)

    # Personalized Recommendations
    st.write("### Personalized Healthcare Recommendations")
    recommendation = df[df["Cluster_GMM"] == selected_cluster]["Recommendation"].iloc[0]
    ml_recommendation = df[df["Cluster_GMM"] == selected_cluster]["ML_Recommendation"].iloc[0]
    st.success(f"Rule-Based: {recommendation}")
    st.info(f"ML-Based: {ml_recommendation}")

    # Feature Importances
    if feature_importances_df is not None:
        st.write("### Feature Importances (ML Recommendation Model)")
        st.dataframe(feature_importances_df.sort_values('Importance', ascending=False))

    # PCA Visualization
    st.write("### PCA: 2D Projection of Clusters")
    if 'PCA1' in df.columns and 'PCA2' in df.columns:
        fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster_GMM', title='PCA Projection (colored by GMM Cluster)')
        st.plotly_chart(fig)
    if pca_explained_var_df is not None:
        st.write("#### PCA Explained Variance Ratio")
        st.dataframe(pca_explained_var_df)

    # Show KMeans Cluster Distribution
    if 'Cluster_KMeans' in df.columns:
        st.write("### KMeans Cluster Distribution")
        fig = px.histogram(df, x="Cluster_KMeans", color="Cluster_KMeans", title="KMeans Cluster Distribution")
        st.plotly_chart(fig)
        # PCA scatter for KMeans
        if 'PCA1' in df.columns and 'PCA2' in df.columns:
            fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster_KMeans', title='PCA Projection (colored by KMeans Cluster)')
            st.plotly_chart(fig)

else:
    st.warning("Required columns ('Cluster_GMM', 'BMI', 'Recommendation') are missing in the data. Please check the backend processing.")