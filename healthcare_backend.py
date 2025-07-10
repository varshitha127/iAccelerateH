import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster
import re
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# Additional imports for profiling and validation
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '_', name)

# List of all CSV files in the data folder
csv_files = [
    'ChronicKidneyDisease_EHRs_from_AbuDhabi.csv',
    'CLEAN- PCOS SURVEY SPREADSHEET.csv',
    'clustered_healthcare_data.csv',
    'Female_Only_Corrected.csv',
    'Female_Only_MentalHealth.csv',
    'kag_risk_factors_cervical_cancer.csv',
    'Maternal Health Risk Data Set.csv',
    'post natal data.csv',
    'reproductiveAgeWomen.csv',
    'survey.csv',
]

# Prepend 'data/' to all file paths
csv_files = [os.path.join('data', f) for f in csv_files]

def convert_range_to_numeric(val):
    if isinstance(val, str):
        match = re.match(r'^(\d+)[-–](\d+)$', val.replace(' ', ''))
        if match:
            return (float(match.group(1)) + float(match.group(2))) / 2
        try:
            return float(val)
        except ValueError:
            return np.nan
    return val

def load_and_merge_csvs(csv_files):
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)  # Track origin
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    merged_df = pd.concat(dfs, axis=0, ignore_index=True, sort=True)
    return merged_df

# Load and merge all data
raw_df = load_and_merge_csvs(csv_files)

# Basic cleaning: drop duplicates, reset index
raw_df = raw_df.drop_duplicates().reset_index(drop=True)

# Select features for clustering (update as needed, exclude 'Cluster_GMM')
# Expanded feature list
possible_features = [
    'Age', 'BMI', 'Blood Pressure', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin',
    'Resistin', 'MCP.1', 'MentalHealthScore', 'Pregnancies',
    # Lifestyle/behavioral
    'Do you eat fast food regularly ?', 'Do you exercise on a regular basis ?', 'Have you gained weight recently?',
    'Feeling sad or Tearful', 'Irritable towards baby & partner', 'Trouble sleeping at night',
    'Problems concentrating or making decision', 'Overeating or loss of appetite', 'Feeling anxious',
    'Feeling of guilt', 'Problems of bonding with baby', 'Suicide attempt',
    # Workplace/mental health
    'work_interfere', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program',
    'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence',
    # Family planning
    'Married or in-union women of reproductive age who have their need for family planning satisfied with modern methods (%)'
]
features = [f for f in possible_features if f in raw_df.columns]

# Convert string ranges to numeric and ensure all features are numeric
def clean_numeric_columns(df, features):
    for col in features:
        df[col] = df[col].apply(convert_range_to_numeric)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

raw_df = clean_numeric_columns(raw_df, features)

# Data validation: Outlier detection, unit checks, impossible value handling
for col in features:
    if raw_df[col].dtype in [float, int]:
        # Remove impossible values (e.g., negative ages, BMI out of range)
        if 'Age' in col:
            raw_df.loc[(raw_df[col] < 0) | (raw_df[col] > 120), col] = np.nan
        if 'BMI' in col:
            raw_df.loc[(raw_df[col] < 10) | (raw_df[col] > 80), col] = np.nan
        # General outlier removal (3-sigma rule)
        mean = raw_df[col].mean()
        std = raw_df[col].std()
        raw_df.loc[(raw_df[col] < mean - 3*std) | (raw_df[col] > mean + 3*std), col] = np.nan
    # For categorical/boolean, replace impossible values with NaN
    if raw_df[col].dtype == object:
        raw_df[col] = raw_df[col].replace(['NA', 'N/A', '', 'Not interested to say', 'Not sure'], np.nan)

# Automated data profiling (Sweetviz for Python 3.13+)
# Remove Sweetviz import and usage

# Manual EDA visualizations
eda_dir = os.path.join('data')
# Correlation heatmap
plt.figure(figsize=(12, 8))
corr = raw_df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'eda_correlation_heatmap.png'))
plt.close()
# Distributions of key features
for col in features:
    if pd.api.types.is_numeric_dtype(raw_df[col]):
        plt.figure(figsize=(8, 4))
        sns.histplot(raw_df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        safe_col = sanitize_filename(col)
        plt.savefig(os.path.join(eda_dir, f'eda_dist_{safe_col}.png'))
        plt.close()
# Missing value matrix
msno.matrix(raw_df[features], figsize=(12, 6))
plt.title('Missing Value Matrix')
plt.tight_layout()
plt.savefig(os.path.join(eda_dir, 'eda_missing_matrix.png'))
plt.close()
# Boxplots for outlier detection
for col in features:
    if pd.api.types.is_numeric_dtype(raw_df[col]):
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=raw_df[col])
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        safe_col = sanitize_filename(col)
        plt.savefig(os.path.join(eda_dir, f'eda_boxplot_{safe_col}.png'))
        plt.close()

# Impute missing values and scale features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X = imputer.fit_transform(raw_df[features])
X_scaled = scaler.fit_transform(X)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
raw_df['PCA1'] = X_pca[:, 0]
raw_df['PCA2'] = X_pca[:, 1]
pd.DataFrame({'PCA_Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))], 'Explained_Variance_Ratio': pca.explained_variance_ratio_}).to_csv(os.path.join('data', 'pca_explained_variance.csv'), index=False)

# Hierarchical Clustering
Z = linkage(X_scaled, method='ward')
raw_df['Cluster_Hier'] = fcluster(Z, t=4, criterion='maxclust')  # 4 clusters as example

# Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=4, random_state=42)
raw_df['Cluster_GMM'] = gmm.fit_predict(X_scaled)

# K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
raw_df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# After clustering
# Calculate evaluation metrics
silhouette_gmm = silhouette_score(X_scaled, raw_df['Cluster_GMM'])
davies_gmm = davies_bouldin_score(X_scaled, raw_df['Cluster_GMM'])
silhouette_hier = silhouette_score(X_scaled, raw_df['Cluster_Hier'])
davies_hier = davies_bouldin_score(X_scaled, raw_df['Cluster_Hier'])
silhouette_kmeans = silhouette_score(X_scaled, raw_df['Cluster_KMeans'])
davies_kmeans = davies_bouldin_score(X_scaled, raw_df['Cluster_KMeans'])

metrics_df = pd.DataFrame({
    'Clustering': ['GMM', 'Hierarchical', 'KMeans'],
    'Silhouette Score': [silhouette_gmm, silhouette_hier, silhouette_kmeans],
    'Davies-Bouldin Index': [davies_gmm, davies_hier, davies_kmeans]
})
metrics_df.to_csv(os.path.join('data', 'clustering_metrics.csv'), index=False)

# Example: Rule-based recommendations per cluster (customize as needed)
def get_recommendation(cluster):
    recs = {
        0: "Focus on regular exercise and balanced diet.",
        1: "Monitor blood pressure and reduce salt intake.",
        2: "Consult a specialist for hormonal balance.",
        3: "Prioritize mental health and stress management.",
    }
    return recs.get(cluster, "General health check-up recommended.")

raw_df['Recommendation'] = raw_df['Cluster_GMM'].apply(get_recommendation)

# Encode recommendations as labels for ML
le = LabelEncoder()
raw_df['Rec_Label'] = le.fit_transform(raw_df['Recommendation'])

# Train/test split (use all data for now, or split if needed)
X_ml = X_scaled
y_ml = raw_df['Rec_Label']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_ml, y_ml)

# Save model and label encoder
joblib.dump(clf, os.path.join('data', 'rf_recommender.joblib'))
joblib.dump(le, os.path.join('data', 'rec_label_encoder.joblib'))

# Feature importances
importances = clf.feature_importances_
importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importances_df.to_csv(os.path.join('data', 'rf_feature_importances.csv'), index=False)

# Predict recommendations for all users
raw_df['ML_Recommendation'] = le.inverse_transform(clf.predict(X_ml))

# Save processed data for frontend use (with ML recommendations)
raw_df.to_csv(os.path.join('data', 'integrated_clustered_healthcare_data.csv'), index=False)

if __name__ == "__main__":
    print("Data integration and clustering complete. Output saved to data/integrated_clustered_healthcare_data.csv.") 