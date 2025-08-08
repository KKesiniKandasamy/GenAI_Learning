# Utilise libraries/Data Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler



# Load the dataset
file_path = "enhanced_diabetes_dataset.csv"
df = pd.read_csv(file_path)
# Define the target variable
target_column = 'Diabetes'

# Split data into features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Normalise the features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Show the first few rows of the result
print("Scaled Features (X):\n", X_scaled.head())
print("\nTarget (y):\n", y.head())