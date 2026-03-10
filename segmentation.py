import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

def load_data(filepath='data/retail_data.csv'):
    df = pd.read_csv(filepath)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def calculate_rfm(df):
    # Calculate Total Price for each transaction line
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Reference date for Recency (usually the day after the last transaction in the dataset)
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Aggregate at Customer level
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    return rfm

def handle_outliers(rfm):
    # Cap values at 99th percentile or use IQR to reduce impact of extreme outliers
    for col in ['Recency', 'Frequency', 'Monetary']:
        q1 = rfm[col].quantile(0.05)
        q3 = rfm[col].quantile(0.95)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # We cap instead of drop to keep all customers
        rfm[col] = np.where(rfm[col] > upper_bound, upper_bound, rfm[col])
        rfm.loc[rfm[col] < lower_bound, col] = lower_bound
    return rfm

def perform_clustering(rfm, n_clusters=3):
    # Scaling is crucial for K-means
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm, kmeans, scaler

def assign_segment_names(rfm):
    # Generic logic to name clusters based on their RFM profile
    cluster_avg = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Simple heuristic:
    # High Monetary, High Frequency -> High Value
    # High Recency (days since last purchase), Low F, Low M -> At Risk / Churned
    # Low Recency, Low Frequency -> New / Emerging
    
    # Sort clusters by Monetary value
    sorted_clusters = cluster_avg.sort_values(by='Monetary').index.tolist()
    
    segment_map = {}
    if len(sorted_clusters) == 3:
        segment_map[sorted_clusters[0]] = 'Low Value / At Risk'
        segment_map[sorted_clusters[1]] = 'Mid Value'
        segment_map[sorted_clusters[2]] = 'High Value / Loyal'
    
    rfm['Segment'] = rfm['Cluster'].map(segment_map)
    return rfm

def visualize_segments(rfm):
    os.makedirs('outputs', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Scatter Plot: Recency vs Monetary
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Segment', palette='viridis', alpha=0.7)
    plt.title('Customer Segments: Recency vs Monetary Value')
    plt.xlabel('Recency (Days since last purchase)')
    plt.ylabel('Monetary Value ($)')
    plt.tight_layout()
    plt.savefig('outputs/scatter_recency_monetary.png')
    plt.close()

    # 2. Scatter Plot: Frequency vs Monetary
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Segment', palette='viridis', alpha=0.7)
    plt.title('Customer Segments: Frequency vs Monetary Value')
    plt.xlabel('Frequency (Number of purchases)')
    plt.ylabel('Monetary Value ($)')
    plt.tight_layout()
    plt.savefig('outputs/scatter_frequency_monetary.png')
    plt.close()
    
    # 3. Boxplots comparing segments
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(ax=axes[0], data=rfm, x='Segment', y='Recency', palette='viridis')
    axes[0].set_title('Recency by Segment')
    sns.boxplot(ax=axes[1], data=rfm, x='Segment', y='Frequency', palette='viridis')
    axes[1].set_title('Frequency by Segment')
    sns.boxplot(ax=axes[2], data=rfm, x='Segment', y='Monetary', palette='viridis')
    axes[2].set_title('Monetary Value by Segment')
    plt.tight_layout()
    plt.savefig('outputs/segment_boxplots.png')
    plt.close()

def main():
    print("Loading data...")
    df = load_data()
    
    print("Calculating RFM...")
    rfm = calculate_rfm(df)
    
    print("Handling outliers...")
    rfm = handle_outliers(rfm)
    
    print("Applying K-Means Clustering...")
    rfm_clustered, model, scaler = perform_clustering(rfm, n_clusters=3)
    
    print("Assigning segment names...")
    rfm_final = assign_segment_names(rfm_clustered)
    
    print("Generating visualizations...")
    visualize_segments(rfm_final)
    
    # Save the clustered output
    rfm_final.to_csv('outputs/customer_segments.csv', index=False)
    print("Segmentation complete. Results saved to 'outputs/customer_segments.csv'")
    
    # Print summary
    summary = rfm_final.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).rename(columns={'CustomerID': 'Count'})
    print("\nSegment Summary:")
    print(summary)


if __name__ == "__main__":
    main()
