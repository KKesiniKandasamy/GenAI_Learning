# Create a Python notebook to implement, train, and evaluate these
# two neural network architectures utilising the CIFAR-10 dataset. 
# Task 1: Utilise Libraries/Dataset 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
x_test = preprocess_input(x_test)
input_shape = x_train.shape[1:]

# Task 2: Generate at least two EDA visualisations
# Visualize some sample images from the dataset
def plot_sample_images(x, y, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i])
        plt.xlabel(class_names[np.argmax(y[i])])
    plt.show()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plot_sample_images(x_train, y_train, class_names)
# Visualize class distribution
def plot_class_distribution(y, class_names):
    plt.figure(figsize=(10, 5))
    sns.countplot(x=np.argmax(y, axis=1))
    plt.xticks(ticks=range(len(class_names)), labels=class_names)
    plt.title('Class Distribution')
    plt.show()
plot_class_distribution(y_train, class_names)

# Task 3: Analyse data quality
# Check for missing values or outliers and data quality issues
def analyze_data_quality(x, y):
    print("Training data shape:", x.shape)
    print("Training labels shape:", y.shape)
    print("Number of classes:", np.unique(np.argmax(y, axis=1)))
    print("Checking for NaN values in images:", np.isnan(x).any())
    print("Checking for NaN values in labels:", np.isnan(y).any())
analyze_data_quality(x_train, y_train)

# Task 4: Construct a CNN model
# Construct a CNN model with appropriate layers (convolutional, pooling, fully connected) 
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
cnn_model = create_cnn_model(input_shape, num_classes)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()

# Task 5: Train the CNN model using the CIFAR-10 dataset
# Train the model using the training dataset with an appropriate number of training epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
start_time = time.time()
cnn_history = cnn_model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping])
end_time = time.time()
print(f"CNN Training Time: {end_time - start_time:.2f} seconds")
# Evaluate the model on the test dataset
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test, y_test)
print(f"CNN Test Accuracy: {cnn_test_acc:.4f}")
# Plot training & validation accuracy and loss values
def plot_training_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
plot_training_history(cnn_history, "CNN")

# Task 6: Display model architecture and training progress
# Display the model architecture and training progress (take a screenshot of the progress for each epoch)
cnn_model.summary()

# Task 7: Construct a ViT model
# Implement a Vision Transformer (ViT) model with attention mechanisms 
def create_vit_model(input_shape, num_classes, patch_size=4, num_heads=4, num_layers=8, dff=128, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    
    # Calculate number of patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patch_dim = patch_size * patch_size * input_shape[2]
    
    # Extract patches using Conv2D and reshape
    patches = layers.Conv2D(patch_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    patches = layers.Reshape((num_patches, patch_dim))(patches)
    
    # Linear projection of patches
    x = layers.Dense(64)(patches)
    
    # Add positional encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=64)(positions)
    x = x + position_embedding
    
    # Transformer blocks
    for _ in range(num_layers):
        x1 = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        ffn_output = Dense(dff, activation='relu')(x3)
        ffn_output = Dense(64)(ffn_output)
        x = layers.Add()([ffn_output, x2])
    
    # Global average pooling and classification
    x = LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
vit_model = create_vit_model(input_shape, num_classes)
vit_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
vit_model.summary()

# Task 8: Train the ViT model using the CIFAR-10 dataset
# Train the model using the training dataset with an appropriate number of training epochs
print("TASK 8: Training ViT Model...")
early_stopping_vit = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
start_time_vit = time.time()
vit_history = vit_model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_val, y_val), callbacks=[early_stopping_vit])
end_time_vit = time.time()
print(f"ViT Training Time: {end_time_vit - start_time_vit:.2f} seconds")
# Evaluate the model on the test dataset
vit_test_loss, vit_test_acc = vit_model.evaluate(x_test, y_test)
print(f"ViT Test Accuracy: {vit_test_acc:.4f}")
# Plot training & validation accuracy and loss values
plot_training_history(vit_history, "ViT")
# Task 9: Display model architecture and training progress
# Display the model architecture and training progress (take a screenshot of the progress for each epoch)
print("Task 9: Display model architecture and training progress")
vit_model.summary()
# Task 10: Compare training and validation results for each model
# Discuss the difference in performance, training efficiency and learning dynamics or learning patterns of each model
print("Task 10: Compare training and validation results for each model")
print("CNN Test Accuracy:", cnn_test_acc)
print("ViT Test Accuracy:", vit_test_acc)
print(f"CNN Training Time: {end_time - start_time:.2f} seconds")
print(f"ViT Training Time: {end_time_vit - start_time_vit:.2f} seconds")
# Generate classification reports and confusion matrices for both models
cnn_y_pred = np.argmax(cnn_model.predict(x_test), axis=1)
vit_y_pred = np.argmax(vit_model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("CNN Classification Report:")
print(classification_report(y_true, cnn_y_pred, target_names=class_names))
print("ViT Classification Report:")
print(classification_report(y_true, vit_y_pred, target_names=class_names))
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()  
plot_confusion_matrix(y_true, cnn_y_pred, class_names, "CNN")
plot_confusion_matrix(y_true, vit_y_pred, class_names, "ViT")   
