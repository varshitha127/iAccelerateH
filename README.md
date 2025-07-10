# iAccelerateH: Personalized Healthcare Recommendations for Women

## Project Overview
This project provides personalized healthcare recommendations for women by integrating multiple health datasets, applying advanced clustering algorithms (Hierarchical Clustering and Gaussian Mixture Models), and delivering actionable insights through an interactive Streamlit dashboard.

## Project Structure

```
iAccelerateH/
│
├── data/                  # All raw and processed CSV files
│   └── [all your CSVs]
│
├── healthcare_backend.py  # Data integration, clustering, recommendations logic
├── Healthcare_Dashboard.py# Streamlit dashboard (frontend)
│
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
```

## Key Components
- data/: Contains all health-related CSV files used for analysis and modeling.
- healthcare_backend.py: Loads, integrates, preprocesses data, applies clustering, and generates recommendations.
- Healthcare_Dashboard.py: Streamlit app for interactive exploration and visualization of clusters and recommendations.
- requirements.txt: Lists all Python dependencies required to run the project.
- README.md: This file. Describes the project, structure, and setup instructions.

## Exploratory Data Analysis (EDA)
The backend automatically generates EDA plots using pandas, seaborn, matplotlib, and missingno:
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
