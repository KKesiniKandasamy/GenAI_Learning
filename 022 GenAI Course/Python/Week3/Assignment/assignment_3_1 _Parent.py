# Utilise libraries/Data Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import sklearn as sk


# Load the dataset
file_path = "enhanced_diabetes_dataset.csv"
df = pd.read_csv(file_path)

# Visualise feature importance using appropriate techniques
# Visualize the distribution of each feature
def plot_feature_distribution(data, feature):
    plt.figure(figsize=(10, 5))
    sns.histplot(data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
# Plot distributions for all numerical features
numerical_features = df.select_dtypes(include='number').columns.tolist()
for feature in numerical_features:
    plot_feature_distribution(df, feature)
# Visualize correlations between features
def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()
plot_correlation_matrix(df)

# Visualize feature importance using a simple model
from sklearn.ensemble import RandomForestClassifier
# Prepare data for feature importance
X = df.drop(columns=['Diabetes'])
y = df['Diabetes']
# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
# Get feature importances
feature_importances = rf_model.feature_importances_
# Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
