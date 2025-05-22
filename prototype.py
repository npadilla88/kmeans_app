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
st.set_page_config(page_title="K-Means Segmentation Web App", layout="wide")
st.markdown(
    """
    <style>
    thead tr th {
        background-color: rgba(200, 16, 46, 1) !important;  /* Full opacity */
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("./img/lbs_logo.jpg", width=100)
st.title("LBS K-Means Segmentation Web App")

# Navigation
st.sidebar.title("Navigation")
sections = ["Instructions", "Data Exploration", "Elbow Plot & Explained Variance", "Segment Descriptions", "Segment Assignments"]
selection = st.sidebar.radio("Go to", sections)

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with ID column and numerical variables", type=["csv"])

# Display instructions
if selection == "Instructions":
    st.markdown("""
    ### Welcome to the London Business School KMeans Segmentation Web App tool for the Marketing Strategy course!

    This web application is designed to assist you in performing KMeans clustering on your dataset. 

    Provided with a CSV file, the app will help you:
                
    - Summarise the data,
    - Determine the optimal number of segments using the Elbow method,
    - Provide detailed descriptions of each cluster given the number of segments you decide.

    ### Instructions
                
    1. **Upload your CSV file**: The first column should contain unique IDs, and the remaining columns should contain numerical variables.
    2. **Select the analysis**: Use the sidebar to navigate through different sections of the app.
    3. **View results**: The app will display summary statistics, the explained variance for different number of segments, cluster descriptions, and cluster assignments.
                
    ### Important Notes
    - Ensure that the first column of your CSV file contains unique IDs.
    - The remaining columns should contain numerical variables suitable for clustering. Those values should belong to comparable scales.         
                
    """)

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
        if selection == "Data Exploration":
            st.write("## Data Exploration")
            st.write("### Raw Data Preview")
            st.dataframe(df)
            st.write("### Summary statistics")
            summary_stats = df.describe().T
            summary_stats['mean'] = summary_stats['mean'].map(lambda x: f"{x:.2f}")
            summary_stats['std'] = summary_stats['std'].map(lambda x: f"{x:.2f}")
            st.dataframe(summary_stats)
        
        # Run K-Means for k = 2 to 8 segments
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
            cluster_assignments[f'Segment_k{k}'] = best_kmeans.labels_ + 1 # Start indexes by one
            
            # Compute Segment descriptions
            cluster_sizes = pd.Series(best_kmeans.labels_).value_counts(normalize=True)
            cluster_means = df_numeric.groupby(best_kmeans.labels_).mean().round(2)
            cluster_description = cluster_means.copy()
            cluster_description.insert(0, "Segment Size", cluster_sizes)
            cluster_description.index = cluster_description.index + 1  # Start indexes by one
            cluster_description.index.name = "Segment"
            cluster_description.reset_index(inplace=True)
            cluster_description.iloc[:, 1:] = cluster_description.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")
            cluster_descriptions[k] = cluster_description
        
        # Elbow Plot
        if selection == "Elbow Plot & Explained Variance":
            explained_variance_percentage = [var * 100 for var in explained_variance]
            fig, ax = plt.subplots(figsize=(8, 5), facecolor='#001E62')
            ax.set_facecolor('#C8102E')
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
            ax.set_xlabel("Number of Segments (k)")
            ax.set_ylabel("Explained Variance (%)")
            ax.set_title("Elbow Method for Optimal k")
            # Display plot and table
            col1, col2 = st.columns([2, 1])
            col1.pyplot(fig)
            explained_variance_percentage = [f"{var * 100:.2f}" for var in explained_variance]
            variance_df = pd.DataFrame({"k": k_values, "Explained Variance (%)": explained_variance_percentage})
            col2.write("### Explained Variance Table")
            col2.dataframe(variance_df)
        
        # Display Segment Descriptions
        if selection == "Segment Descriptions":
            st.write("### Segment Description Table")
            k_options = list(k_values)
            selected_k = st.selectbox("Select the number of segments (k)", k_options)
            if selected_k in cluster_descriptions:
                st.write(f"#### Segment Description for k = {selected_k}")
                st.dataframe(cluster_descriptions[selected_k])
                # Extract the selected segment table
                df_seg = cluster_descriptions[selected_k]
                # Transform all columns to numeric
                df_seg.iloc[:, 1:] = df_seg.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
                # Get list of numeric columns to choose from (excluding 'Segment' and 'Segment Size')
                available_columns = df_seg.columns.drop(["Segment", "Segment Size"], errors='ignore').tolist()
                # Let user select X and Y axes
                default_x = 0 if len(available_columns) > 0 else 0
                default_y = 1 if len(available_columns) > 1 else default_x
                x_col = st.selectbox("Select X-axis variable", available_columns, index=default_x)
                y_col = st.selectbox("Select Y-axis variable", available_columns, index=default_y)
                # Original data for defining the plot limits
                # Plotting
                x_min, x_max = df[x_col].min() - 0.5, df[x_col].max() + 0.5
                y_min, y_max = df[y_col].min() - 0.5, df[y_col].max() + 0.5
                fig, ax = plt.subplots(figsize=(8, 8), facecolor='#001E62')
                ax.set_facecolor('#001E62')
                # Plot each segment as a bubble
                sizes = pd.to_numeric(df_seg["Segment Size"], errors="coerce").fillna(0) * 3000
                ax.scatter(
                    df_seg[x_col],
                    df_seg[y_col],
                    s=sizes,  # Scale bubble size
                    # alpha=0.6,
                    c='#C8102E',
                    edgecolors='white'
                )
                # Annotate with segment labels
                for idx, row in df_seg.iterrows():
                    ax.text(row[x_col], row[y_col], str(int(row["Segment"])), fontsize=10, ha='center', va='center', color='white')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_title(f"Segment Description for k = {selected_k}")
                ax.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig)
            else:
                st.error("Invalid selection. Please select a valid number of segments.")
        
        # Display Segment Assignments
        if selection == "Segment Assignments":
            st.write("### Segment Assignments")
            st.dataframe(cluster_assignments)
