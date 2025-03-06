import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load your clustered dataset
dataset="clustered_healthcare_data.csv"
df = pd.read_csv(dataset)

# Streamlit App
st.title("Personalized Healthcare Dashboard")

# Sidebar Filters
st.sidebar.title("User Filters")
selected_cluster = st.sidebar.selectbox("Select Cluster:", df["Cluster_GMM"].unique())

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
st.success(f"Recommended Action: {recommendation}")
