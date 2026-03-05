import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans

def get_wcss_scores(df, max_k=10):
    """
    Computes WCSS for K values 1 through max_k. 
    Used to mathematically find the 'Elbow'.
    """
    # Use only LAT and LON for geospatial density
    coords = df[['LAT', 'LON']]
    wcss = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(coords)
        wcss.append(kmeans.inertia_) # inertia_ is the WCSS
        
    return wcss

def apply_optimized_kmeans(df, k_value):
    """
    Applies KMeans using the optimal K found in the elbow plot.
    """
    coords = df[['LAT', 'LON']]
    kmeans = KMeans(n_clusters=k_value, n_init=10, random_state=42)
    df['cluster_id'] = kmeans.fit_predict(coords)
    return df, kmeans

def find_mathematical_elbow(wcss_list):
    """
    Automatically detects the 'bend' in the elbow curve using 
    the maximum curvature method.
    """
    # Calculate the slopes between points
    slopes = [wcss_list[i] - wcss_list[i+1] for i in range(len(wcss_list)-1)]
    
    # The 'elbow' is often where the drop in error decreases by more than 50%
    for i in range(len(slopes)-1):
        if slopes[i+1] < (slopes[i] * 0.5):
            return i + 2 # +2 to account for index and start at K=1
    return 4 # Default fallback

def identify_optimal_k(df, max_k=10):
    """
    Mathematically identifies the 'Elbow' point by finding the 
    maximum curvature in the WCSS graph.
    """
    coords = df[['LAT', 'LON']]
    wcss = []
    
    # 1. Calculate WCSS for a range of K
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(coords)
        wcss.append(kmeans.inertia_)
    
    # 2. Find the 'Knee' (The Elbow) using the second derivative/acceleration method
    # This identifies where the WCSS curve 'bends' most sharply
    x = np.arange(1, len(wcss) + 1)
    y = np.array(wcss)
    
    # Calculate the first and second derivatives of the curve
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    
    # The elbow is typically at the index of the maximum second derivative (the peak of the bend)
    optimal_k = x[np.argmax(d2y)]
    
    return int(optimal_k), wcss

def generate_k_comparison_data(df, k_list=[2, 4]):
    """
    Generates multiple versions of the dataset with different K values 
    for comparison in the predictive phase.
    """
    results = {}
    for k in k_list:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        # Create a copy so we don't overwrite the original dataframe columns
        df_temp = df.copy()
        df_temp['cluster_id'] = kmeans.fit_predict(df_temp[['LAT', 'LON']])
        results[k] = df_temp
        print(f"Generated Clustered Data for K={k}")
        
    return results

def get_tournament_k_values(df, max_k=10):
    """
    Identifies the mathematical elbow (Point 1) and the next 
    significant stability point (Point 2) for comparison.
    """
    coords = df[['LAT', 'LON']]
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(coords)
        wcss.append(kmeans.inertia_)
    
    # Calculate acceleration (second derivative) to find the 'bend'
    x = np.arange(1, len(wcss) + 1)
    dy = np.gradient(wcss, x)
    d2y = np.gradient(dy, x)
    
    # Point 1: The primary mathematical elbow (The Knee)
    elbow_k = int(x[np.argmax(d2y)])
    
    # Point 2: An operational candidate (usually Elbow + 2 for granularity)
    candidate_k = elbow_k + 2
    
    return [elbow_k, candidate_k], wcss

def apply_k_tournament(df, k_list):
    """Generates datasets for all identified K-values in the tournament."""
    datasets = {}
    for k in k_list:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        df_temp = df.copy()
        df_temp['cluster_id'] = kmeans.fit_predict(df_temp[['LAT', 'LON']])
        datasets[k] = df_temp
    return datasets

def print_cluster_summary(df, label):
    print(f"\n{label} Statistics")
    # Grouping by cluster to see the 'weight' of each zone
    summary = df.groupby('cluster_id').size().reset_index(name='Total_Crimes')
    summary['Percentage'] = (summary['Total_Crimes'] / len(df) * 100).round(2)
    print(summary)
