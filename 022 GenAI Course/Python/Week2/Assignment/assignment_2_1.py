# Utilise Libraries/Data Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = "customer_data.csv"
df = pd.read_csv(file_path)
print("\nDataFrame loaded from CSV:\n", df)

# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:\n", df.head())
# Show basic information and statistics of the DataFrame
print("\nDataFrame info:\n", df.info())
# Show the statistics of the DataFrame
print("\nDataFrame description:\n", df.describe())

# DAta Visualization and enhancement
# =========================================
# Scatter plot of Annual Income vs Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Age', palette='viridis')
plt.title('Annual Income vs Spending Score Scatter Plot', fontweight='bold', color='red')
plt.xlabel('Annual Income', fontweight='bold', color='blue')
plt.ylabel('Spending Score', fontweight='bold', color='blue')
plt.show()

# Correlation heatmap to understand relationships
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Create distribution plots for all variables
for column in df.select_dtypes(include=[np.number]).columns:
    if column.lower() == 'customerid' or column.lower() == 'customer_id':
        continue  # Skip this column
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Check for missing values and outliers
# check for missing values and handle them appropriately
print("\nMissing values in the DataFrame:\n", df.isnull().sum())
# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)
# Save the cleaned DataFrame to a new CSV file
output_file_path = "cleaned_customer_data.csv"
df.to_csv(output_file_path, index=False)
print(f"\nCleaned DataFrame saved to {output_file_path}")



# Outlier detection using boxplots and decide whether to remove them
# Skip customer ID column for outlier detection
num_cols = df.select_dtypes(include=[np.number]).columns.drop('CustomerID')
plt.figure(figsize=(10, 6)) 
sns.boxplot(data=df[num_cols], palette='Set2', orient='h')
plt.title('Boxplot of Numerical Features', fontweight='bold', color='purple')
plt.xticks(rotation=45)
plt.show()

# Apply standardisation or min-max scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)
print("\nScaled DataFrame:\n", df_scaled.head())

## Determine optimal number of clusters using the Elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

# Calculate the silhouette score for different values of k
from sklearn.metrics import silhouette_score
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid()
plt.show()

#Determine the  best value of k
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal number of clusters determined by silhouette score: {optimal_k}")

# Initialise and train the K-Means model with the optimal k value and assign cluster labels to each customer
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_
print("\nDataFrame with Cluster Labels:\n", df.head())
# Visualise the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1', style='Cluster', s=100)
plt.title('Customer Segmentation Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
# Save the clustered DataFrame to a new CSV file
output_file_path = "customer_clusters.csv"
df.to_csv(output_file_path, index=False)
print(f"\nClustered DataFrame saved to {output_file_path}")
# Save the scaled DataFrame to a new CSV file