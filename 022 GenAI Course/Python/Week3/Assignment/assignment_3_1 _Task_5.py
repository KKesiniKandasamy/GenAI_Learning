import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load your data
df = pd.read_csv("enhanced_diabetes_dataset.csv")
target_column = 'Diabetes'

# Split into X and y
X = df.drop(columns=[target_column])
y = df[target_column]

# Scale features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Step 1: Handle Class Imbalance with SMOTE
print("Original class distribution:", Counter(y))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Resampled class distribution:", Counter(y_resampled))

# Step 2: Split into Train (70%), Validation (15%), Test (15%)

# First, split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.30, random_state=42, stratify=y_resampled)

# Now split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Output the shape of each set
print("Train set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

# Select appropriate loss function and evaluation metrics
loss_function = 'binary_crossentropy'  # For binary classification
evaluation_metrics = ['accuracy', 'precision', 'recall', 'f1-score']
# Note: The actual implementation of the model training and evaluation would follow here.
# This code snippet prepares the data for training a machine learning model, ensuring that the dataset is balanced and split into appropriate sets for training, validation, and testing.
# Show the first few rows of the result
print("Scaled Features (X):\n", X_scaled.head())
print("\nTarget (y):\n", y.head())
