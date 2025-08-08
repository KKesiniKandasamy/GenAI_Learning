## ==========================================================
# TASK1:  Import libraries
## ==========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import sklearn as sk
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "enhanced_diabetes_dataset.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print("Initial DataFrame:\n", df.head())
# Show basic information about the DataFrame
print("\nDataFrame Info:")
df.info()
# Show summary statistics of the DataFrame
print("\nSummary Statistics:\n", df.describe())

## ==========================================================
### TASK 2: Generate at least three EDA visualisations
## ==========================================================
# Create distribution plots comparing features between diabetic and non-diabetic patients
def plot_feature_distribution(data, feature):
    plt.figure(figsize=(10, 5))
    sns.histplot(data=data, x=feature, hue='Diabetes', kde=True, stat="density", common_norm=False, palette="Set2")
    plt.title(f'Distribution of {feature} by Diabetes Status')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend(title='Diabetes', loc='upper right')
    plt.show()
# Develop correlation heatmap to identify relationships between features
def plot_correlation_matrix(data):
    plt.figure(figsize=(12, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix')
    plt.show()
# Visualise feature importance using appropriate techniques
def plot_feature_importance(data, target):
    from sklearn.ensemble import RandomForestClassifier
    X = data.drop(columns=[target])
    y = data[target]
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, y)
    feature_importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance from Random Forest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
# Call the plotting functions for EDA
plot_feature_distribution(df, 'Glucose')
plot_feature_distribution(df, 'BMI')
plot_feature_distribution(df, 'Age')
plot_feature_distribution(df, 'Gender')
plot_feature_distribution(df, 'Smoking')
plot_feature_distribution(df, 'BloodPressure')
plot_feature_distribution(df, 'SkinThickness')
plot_correlation_matrix(df)
plot_feature_importance(df, 'Diabetes')

## ==========================================================
# TASK 3: Analyse data quality
## ==========================================================
# Check for missing or zero values (particularly in SkinThickness, Insulin, and BMI)
def check_missing_and_zero_values(data):
    missing_values = data.isnull().sum()
    zero_values = (data == 0).sum()
    return missing_values, zero_values
missing_values, zero_values = check_missing_and_zero_values(df)
print("\nMissing Values:\n", missing_values)
print("\nZero Values:\n", zero_values)

# Identify and handle outliers
def detect_outliers_iqr(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
# Create a new DataFrame to store outliers
outliers = {}
features = df.select_dtypes(include='number').columns.tolist()
for feature in features:
    outliers[feature] = detect_outliers_iqr(df, feature)
# Display outliers for each feature
for feature, outlier_data in outliers.items():
    if not outlier_data.empty:
        print(f"\nOutliers detected in {feature}:\n", outlier_data)
    else:
        print(f"No outliers detected in {feature}.")

## ==========================================================
# TASK 4:  Prepare Features and Target Variable
## ==========================================================
# Split data into features (X) and target (y)
target_column = 'Diabetes'
X = df.drop(columns=[target_column])
y = df[target_column]
# Normalise/standardise the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# Display the first few rows of the scaled features
print("\nScaled Features (X):\n", X_scaled.head())
# Display the target variable
print("\nTarget Variable (y):\n", y.head())


## ==========================================================
# TASK 5: Class Imbalance and Dataset Splitting
## ==========================================================
# Handle class imbalance if present using class weights. if not present, print not present
from collections import Counter
print("\nOriginal class distribution:", Counter(y))
if len(Counter(y)) > 2:
    print("Class imbalance detected. Applying SMOTE...")
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    print("Resampled class distribution:", Counter(y_resampled))
else:
    print("No class imbalance detected. Proceeding without SMOTE.")
#Split data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled if 'X_resampled' in locals() else X_scaled, 
    y_resampled if 'y_resampled' in locals() else y, 
    test_size=0.30, random_state=42, stratify=y_resampled if 'y_resampled' in locals() else y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Output the shape of each set
print("\nTrain set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

## ==========================================================
# TASK6: Design an Appropriate DCNN-based Architecture
## ==========================================================
print("\nDesigning the DCNN-based architecture...Not suitable for this dataset. DCNN is typically used for image data, while this dataset is tabular. Proceeding with a simple neural network instead.")
# Define a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.5),  # Dropout layer for regularisation
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),  # L2 regularisation
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])
# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Display the model summary
print("\nModel Summary ( TASK 6):")
print(model.summary())

## ==========================================================
# TASK 7: Configure the Training Process
## ==========================================================
print("\nConfiguring the training process.")
# Select appropriate loss function and evaluation metrics
loss_function = 'binary_crossentropy'  # For binary classification
evaluation_metrics = ['accuracy', 'precision', 'recall']
# Choose optimiser and learning rate
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
# Implement regularisation techniques (dropout, L1/L2) to prevent overfitting
from tensorflow.keras import regularizers
# Compile the model with the selected loss function, optimizer, and metrics
model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=evaluation_metrics
)
# Display the model summary after configuration
print("\nModel Summary (after configuration):")
print(model.summary())

## =======================================================================
# TASK 8: Train the model with Appropriate Batch Size and Number of Epochs
## =======================================================================
print("\nTraining the model with appropriate batch size and number of epochs.")
# Set batch size and number of epochs
batch_size = 32
epochs = 50
evaluation_metrics = ['accuracy']
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)
# Display training history
print("\nTraining History:")
print(history.history)

## ==========================================================
# TASK 9: Implement Callbacks
## ==========================================================
# Early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping],
    verbose=1

    
)
# Model checkpointing to save the best model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
) 
# Train the model with both early stopping and checkpointing
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)   

## ==========================================================
# TASK 10: Visualise the training process
## ==========================================================
# Plot training and validation loss
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
plot_training_history(history)
# Plot training and validation accuracy
def plot_accuracy(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
plot_accuracy(history)
# Analyse convergenc and potential overfitting/underfitting
def analyze_convergence(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    if val_loss[-1] < min(loss):
        print("Model is converging well.")
    else:
        print("Model may not be converging properly.")

    if max(accuracy) - max(val_accuracy) > 0.1:
        print("Potential overfitting detected.")
    elif max(val_accuracy) < 0.7:
        print("Model may be underfitting.")
    else:
        print("Model performance is acceptable.")   
analyze_convergence(history)

## ==========================================================
# TASK 11: Tune Hyperparameters
## ==========================================================
# Adjust learning rate, batch size or network architecture. use values learning_rate=[0.01, 0.001], batch_size=[16,32], dropout_rate=[0.2], filter_number=[32, 64], kernel_size=[3]
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    from sklearn.model_selection import ParameterGrid
    param_grid = {
        'learning_rate': [0.01, 0.001],
        'batch_size': [16, 32],
        'dropout_rate': [0.2],
        'filter_number': [32, 64],
        'kernel_size': [3]
    }
    grid = ParameterGrid(param_grid)
    
    best_accuracy = 0
    best_params = None
    
    for params in grid:
        print(f"Training with parameters: {params}")
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dropout(params['dropout_rate']),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=params['batch_size'],
            verbose=1
        )
        
        val_accuracy = max(history.history['val_accuracy'])
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params
            
    print(f"Best parameters: {best_params} with validation accuracy: {best_accuracy}")

tune_hyperparameters(X_train, y_train, X_val, y_val)

## ==========================================================
# TASK 12: Evaluate the Model on the Test Set
## ==========================================================
print("\nEvaluating the model on the test set.")
# Calculate accuracy, precision, recall, F1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nTest Set Evaluation:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}")

# Generate confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Create and analyse ROC curve and calculate AUC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
# Save the model
model.save('diabetes_prediction_model.h5')

##### End of the script #####