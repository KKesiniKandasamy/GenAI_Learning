# Utilise libraries/Data Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import sklearn as sk

import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.withdraw()  # Hide the root window

messagebox.showinfo("Task1", "Utilise Libraries/Dataset")


# Load the dataset
file_path = "enhanced_diabetes_dataset.csv"
df = pd.read_csv(file_path)
# Display the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:\n", df.head())

# Show basic information of the DataFrame
print("\nDataFrame info:\n", df.info())
# Show the statistics of the DataFrame
print("\nDataFrame description:\n", df.describe())

# Create distribution plots comparing diabetes and non-diabetes patients
print("\nCreating distribution plots comparing features between diabetic and non-diabetic patients:")
# Create plots
# Automatically select numerical columns (excluding Diabetes itself from the loop)
features = df.select_dtypes(include='number').columns.tolist()
features = [f for f in features if f != 'Diabetes']  # exclude target column

# Set visual style
sns.set(style="whitegrid")

# Create histogram + KDE plots by Diabetes status
for feature in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x=feature,
        hue="Diabetes",
        kde=True,
        stat="density",
        common_norm=False,
        palette="Set2"
    )
    plt.title(f"Distribution of {feature} by Diabetes Status")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend(title="Diabetes", labels=["Non-Diabetic", "Diabetic"])
    plt.tight_layout()
    plt.show()

## Correlation Heatmap
    # Select only numerical features
numerical_df = df.select_dtypes(include='number')

# Compute the correlation matrix
corr_matrix = numerical_df.corr()

# Set the figure size and style
plt.figure(figsize=(14, 10))
sns.set(style="white")

# Draw the heatmap
sns.heatmap(
    corr_matrix,
    annot=True,          # Show correlation values
    fmt=".2f",           # Format decimal places
    cmap="coolwarm",     # Color palette
    center=0,            # Center at 0
    linewidths=0.5,      # Add lines between cells
    square=True          # Make cells square
)

plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# Check for missing or zero values (particularly in SkinThickness, Insulin, and BMI)
missing_values = df.isnull().sum()
print("\nMissing values in each column:\n", missing_values)
# Check for zero values in specific columns
zero_values = (df[['SkinThickness', 'Insulin', 'BMI']] == 0).sum()
print("\nZero values in SkinThickness, Insulin, and BMI:\n", zero_values)

# Identify and handle outliers
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
# Detect outliers for each feature and update in a doc file

outliers = {}
for feature in features:
    outliers[feature] = detect_outliers_iqr(df, feature)   
# Print outliers for each feature
for feature, outlier_data in outliers.items():
    if not outlier_data.empty:
        print(f"\nOutliers detected in {feature}:\n", outlier_data)
    else:
        print(f"\nNo outliers detected in {feature}.")
# Handle outliers by removing them
for feature in features:
    outlier_indices = detect_outliers_iqr(df, feature).index
    df = df.drop(index=outlier_indices) 
# Reset index after dropping outliers
df.reset_index(drop=True, inplace=True)
# Display the cleaned DataFrame
print("\nDataFrame after removing outliers:\n", df.head())

# Prepare Features and Target Variable
X = df.drop(columns=['Diabetes'])
y = df['Diabetes']
# Display the first few rows of the cleaned DataFrame
print("\nFirst few rows of the cleaned DataFrame:\n", df.head())
# Show basic information of the cleaned DataFrame
print("\nCleaned DataFrame info:\n", df.info())
# Show the statistics of the cleaned DataFrame
print("\nCleaned DataFrame description:\n", df.describe())
# Save the cleaned DataFrame to a new CSV file
df.to_csv("cleaned_enhanced_diabetes_dataset.csv", index=False)
# Save the cleaned DataFrame to a new CSV file
print("\nCleaned DataFrame saved to 'cleaned_enhanced_diabetes_dataset.csv'.")
# Prepare Features and Target Variable
X = df.drop(columns=['Diabetes'])
y = df['Diabetes']
# Display the first few rows of the cleaned DataFrame
print("\nFirst few rows of the cleaned DataFrame:\n", df.head())
# Show basic information of the cleaned DataFrame
print("\nCleaned DataFrame info:\n", df.info())
# Show the statistics of the cleaned DataFrame
print("\nCleaned DataFrame description:\n", df.describe())
# Save the cleaned DataFrame to a new CSV file
df.to_csv("cleaned_enhanced_diabetes_dataset.csv", index=False)
print("\nCleaned DataFrame saved to 'cleaned_enhanced_diabetes_dataset.csv'.")

# configure the training process
from sklearn.preprocessing import MinMaxScaler
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

# Train the model with Appropriate Batch Size and Number of Epochs
from sklearn.model_selection import train_test_split
# First, split into train (70%) and temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y)
# Now split temp into validation (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
# Output the shape of each set
print("Train set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Test set:", X_test.shape)

#Implement Callbacks for Early Stopping and Model Checkpointing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True
)

print ("Task 7 Confgire the training process")

# Select appropriate loss function and evaluation metrics
loss_function = 'binary_crossentropy'  # For binary classification
evaluation_metrics = ['accuracy', 'precision', 'recall', 'f1-score']    
# Choose optimiser and learning rate
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
# Implement regularisation techniques ( dropout, L1/L2 regularisation) to prevent overfitting
from tensorflow.keras import regularizers
# Define a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.5),  # Dropout layer for regularisation
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 regularisation
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])
# Compile the model
model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=evaluation_metrics
)

# Implement callbacks for early stopping and model checkpointing
callbacks = [early_stopping, model_checkpoint]
# Show the model summary
print("\nModel Summary:")
model.summary()
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)
# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(
    X_test, y_test, verbose=0
)   
print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("Test F1-Score:", test_f1)

# Learning rate scheduling
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)
# Add the learning rate scheduler to callbacks
callbacks.append(reduce_lr)
# Retrain the model with the new callbacks
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)
# Evaluate the model again on the test set
test_loss, test_accuracy, test_precision, test_recall, test_f1 = model.evaluate(
    X_test, y_test, verbose=0
)
print("\nTest Loss after retraining:", test_loss)
print("Test Accuracy after retraining:", test_accuracy)
print("Test Precision after retraining:", test_precision)
print("Test Recall after retraining:", test_recall)
print("Test F1-Score after retraining:", test_f1)


# Plot training history
plt.figure(figsize=(12, 6)) 
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Save the trained model
model.save('diabetes_prediction_model.h5')  
# Analyse convergence and potential overfitting and underfitting
if history.history['loss'][-1] < 0.1 and history.history['val
_loss'][-1] < 0.1:
    print("\nModel converged successfully.")
else:
    print("\nModel did not converge, consider adjusting hyperparameters or model architecture.")   
# Check for overfitting
if history.history['loss'][-1] < history.history['val_loss'][-1]:
    print("\nModel may be overfitting, consider adding more regularisation or reducing model complexity.")
else:
    print("\nModel is not overfitting, training and validation losses are comparable.")
# Show the first few rows of the result
print("Scaled Features (X):\n", X_scaled.head())
print("\nTarget (y):\n", y.head())

# Tune Hyperparameters
from sklearn.model_selection import GridSearchCV
# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.001],
    'batch_size': [16, 32],
    'dropout_rate': [0.2],
    'filter_numbers': [32, 64],
    'kernel_size': [3]
}
# Adjust the model to accept hyperparameters
def create_model(learning_rate=0.001, dropout_rate=0.5, filter_numbers=64, kernel_size=3):
    model = keras.Sequential([
        keras.layers.Dense(filter_numbers, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Calculate accuracy, precision, recall, and F1-score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score)
}
# Create a KerasClassifier for GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, verbose=0)
# Perform Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, refit='accuracy', cv=3)
grid_result = grid.fit(X_train, y_train)
# Print the best hyperparameters and scores
print("\nBest Hyperparameters:")
print(grid_result.best_params_)
print("\nBest Scores:")
print("Accuracy:", grid_result.best_score_)
# Evaluate the best model on the test set