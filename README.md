<<<<<<< HEAD
Personalized Healthcare Recommendations for Women Using Clustering

📌 Project Overview

This project aims to provide personalized healthcare recommendations for women by using Hierarchical Clustering and Gaussian Mixture Models (GMMs) to segment individuals based on health needs and preferences. The final output is an interactive Streamlit dashboard that allows users to explore recommendations based on their health data.

🚀 FEATURES

Data Preprocessing & Clustering: Cleans and prepares healthcare data for clustering.

Machine Learning Models: Implements Hierarchical Clustering and Gaussian Mixture Models (GMMs).

Personalized Recommendations: Provides tailored healthcare insights based on clusters.

Interactive Dashboard: Built with Streamlit and Plotly for visualization.

Deployment: Can be deployed using Streamlit Cloud or local execution.


📁 iAccelerateH 
│── Healthcare_dashboard.py # Streamlit dashboard          
│── recommendation.py       # Personalized recommendations logic # Clustering models (GMM, Hierarchical)
│── requirements.txt        # Required Python libraries
│── README.md               # Project documentation
│── data
│   ├── clustered_healthcare_data.csv # Sample dataset
=======
# Personalized Healthcare Dashboard

## Project Overview
This project provides personalized healthcare recommendations for women by integrating multiple health datasets, applying advanced clustering algorithms (Hierarchical Clustering and Gaussian Mixture Models), and delivering actionable insights through an interactive Streamlit dashboard.

## Project Structure

```
InfosysHacethon/
│
├── data/                        # All raw and processed CSV files
│   └── [all your CSVs]
│
├── healthcare_backend.py        # Data integration, clustering, recommendations logic
├── Healthcare_Dashboard.py      # Streamlit dashboard (frontend)
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview and instructions
│
└── (Optional: Add more folders as you expand)
    ├── models/                  # For saving/loading trained models
    ├── utils/                   # Utility scripts/functions
    └── notebooks/               # Jupyter notebooks for exploration
```

## Key Components
- **data/**: Contains all health-related CSV files used for analysis and modeling.
- **healthcare_backend.py**: Loads, integrates, preprocesses data, applies clustering, and generates recommendations.
- **Healthcare_Dashboard.py**: Streamlit app for interactive exploration and visualization of clusters and recommendations.
- **requirements.txt**: Lists all Python dependencies required to run the project.
- **README.md**: This file. Describes the project, structure, and setup instructions.

## Exploratory Data Analysis (EDA)

The backend automatically generates manual EDA plots using pandas, seaborn, matplotlib, and missingno:
- Correlation heatmap
- Distributions of key features
- Missing value matrix
- Boxplots for outlier detection

All plots are saved as PNG files in the `data/` directory for easy review.

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- missingno

(You do not need ydata-profiling or Sweetviz; EDA is handled manually for Python 3.13+ compatibility.)

## Setup Instructions
1. Clone the repository or download the project files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend script to process and cluster the data:
   ```bash
   python healthcare_backend.py
   ```
4. Launch the Streamlit dashboard:
   ```bash
   streamlit run Healthcare_Dashboard.py
   ```

## Expansion Ideas
- Add more health domains or datasets to `data/`.
- Modularize code further (e.g., move clustering or recommendation logic to `utils/`).
- Save/load trained models in `models/`.
- Use Jupyter notebooks in `notebooks/` for data exploration.

---
For any questions or contributions, please open an issue or pull request. 
>>>>>>> 531fcb7 (Update EDA, dependencies, and documentation for Python 3.13+ compatibility)
