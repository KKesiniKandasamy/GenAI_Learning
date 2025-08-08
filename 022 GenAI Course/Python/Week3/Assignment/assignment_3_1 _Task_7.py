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


print ("Task 7 Confgire the training process - Tria 2")

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

