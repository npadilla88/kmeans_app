import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from io import StringIO

# Set random seed for reproducibility
RANDOM_SEED = 42

# Set a dark background style
plt.style.use('dark_background')


# Streamlit UI
st.set_page_config(page_title="K-Means Clustering Web App", layout="wide")
st.title("K-Means Clustering Web App")
st.write("Upload a CSV file with ID column and numerical variables.")

# Navigation
st.sidebar.title("Navigation")
sections = ["Raw Data", "Elbow Plot & Explained Variance", "Cluster Descriptions", "Cluster Assignments"]
selection = st.sidebar.radio("Go to", sections)

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Check if the first column is ID and rest are numerical
    if df.shape[1] < 2:
        st.error("The file must have an ID column and at least one numerical variable.")
    else:
        id_col = df.columns[0]
        numeric_cols = df.columns[1:]
        df_numeric = df[numeric_cols]
        
        # Display raw data
        if selection == "Raw Data":
            st.write("### Raw Data Preview")
            st.dataframe(df)
            st.write("### Summary statistics")
            summary_stats = df.describe().T
            summary_stats['mean'] = summary_stats['mean'].map(lambda x: f"{x:.2f}")
            summary_stats['std'] = summary_stats['std'].map(lambda x: f"{x:.2f}")
            st.dataframe(summary_stats)
        
        # Run K-Means for k = 2 to 8 clusters
        k_values = list(range(2, 9))
        inertia_values = []
        explained_variance = []
        cluster_assignments = pd.DataFrame({id_col: df[id_col]})
        cluster_descriptions = {}
        total_variance = np.sum(np.square(df_numeric - df_numeric.mean()).values)

        for k in k_values:
            best_kmeans = None
            best_inertia = float('inf')
            
            # Run multiple initializations and choose the best one
            for _ in range(10):  # Running KMeans 10 times
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
                kmeans.fit(df_numeric)
                
                if kmeans.inertia_ < best_inertia:
                    best_inertia = kmeans.inertia_
                    best_kmeans = kmeans
            
            inertia_values.append(best_inertia)
            explained_variance.append(1 - (best_inertia / total_variance))
            
            # Store cluster assignments
            cluster_assignments[f'Cluster_k{k}'] = best_kmeans.labels_ + 1 # Start indexes by one
            
            # Compute cluster descriptions
            cluster_sizes = pd.Series(best_kmeans.labels_).value_counts(normalize=True)
            cluster_means = df_numeric.groupby(best_kmeans.labels_).mean().round(2)
            cluster_description = cluster_means.copy()
            cluster_description.insert(0, "Cluster Size", cluster_sizes)
            cluster_description.index = cluster_description.index + 1  # Start indexes by one
            cluster_description.index.name = "Cluster"
            cluster_description.reset_index(inplace=True)
            cluster_description.iloc[:, 1:] = cluster_description.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")
            cluster_descriptions[k] = cluster_description
        
        # Elbow Plot
        if selection == "Elbow Plot & Explained Variance":
            explained_variance_percentage = [var * 100 for var in explained_variance]
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='#C8102E')
            ax.plot(
                k_values,
                explained_variance_percentage,
                marker='o',
                linestyle='--',
                # color='#EBE8E5',        # light primary color for contrast
                color='#FFFFFF',        # light primary color for contrast
                linewidth=2,
                markersize=6
            )            
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Explained Variance (%)")
            ax.set_title("Elbow Method for Optimal k")
            # Display plot and table
            col1, col2 = st.columns([2, 1])
            col1.pyplot(fig)
            explained_variance_percentage = [f"{var * 100:.2f}" for var in explained_variance]
            variance_df = pd.DataFrame({"k": k_values, "Explained Variance (%)": explained_variance_percentage})
            col2.write("### Explained Variance Table")
            col2.dataframe(variance_df)
        
        # Display Cluster Descriptions
        if selection == "Cluster Descriptions":
            st.write("### Cluster Description Table")
            for k, desc_df in cluster_descriptions.items():
                st.write(f"#### Cluster Description for k = {k}")
                st.dataframe(desc_df)
        
        # Display Cluster Assignments
        if selection == "Cluster Assignments":
            st.write("### Cluster Assignments")
            st.dataframe(cluster_assignments)
