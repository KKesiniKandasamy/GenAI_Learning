#  Train, and evaluate two neural network architectures (basic and advanced Transformer models), and compare their performance on sentiment classification using appropriate metrics.
# ===============================================================================================
# Task 1: Utilise Libraries/Dataset
# ===============================================================================================
# Import the necessary libraries (TensorFlow, datasets, Keras, pandas, Matplotlib, sklearn, etc.)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


#  Load the IMDB dataset using load_dataset("imdb")
print ("==============================================================================")
print("Task 1: Loading the IMDB dataset...")
print ("==============================================================================")
from datasets import load_dataset
dataset = load_dataset("imdb")
print ("==============================================================================")
print("Task 1: Converting dataset to pandas DataFrame...")
print ("==============================================================================")
# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset['train'])
# Print sample sizes
print ("==============================================================================")
print("Tasl 1: Sample sizes:")
print ("==============================================================================")
print(f"Training samples: {len(df)}")
print(f"Testing samples: {len(dataset['test'])}")

# ===============================================================================================
# Task 2: Data Processing and Exploration
# ===============================================================================================
# Generate two EDA visualisations (e.g., review length distribution, class distribution)
print ("==============================================================================")
print("Task 2: Data Processing and Exploration...")
print ("==============================================================================")
# Add a column for review length
df['review_length'] = df['text'].apply(len)
# Plot review length distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title('Review Length Distribution')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()
# Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title('Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.show()
# Check for missing values and data quality
print ("==============================================================================")
print("Task 2: Checking for missing values and data quality...")
print ("==============================================================================")
print(df.isnull().sum())
print(df.duplicated().sum())

# Normalise and tokenise text, pad sequences to a consistent length
print ("================================================================================================")
print("Task 2: Normalising, tokenising text, and padding sequences & convert text to appropriate format")
print ("=================================================================================================")
max_words = 10000
max_len = 256
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, df['label'], test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation labels shape: {y_val.shape}")



# (Already done above with max_len=200)
print(f"Sample tokenized and padded sequence: {X_train[0]}")
print(f"Corresponding label: {y_train.iloc[0]}")
# Create training and testing splits
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Training labels: {len(y_train)}")
print(f"Validation labels: {len(y_val)}")
print ("==============================================================================")
print("Task 2: Data processing complete.")
print ("==============================================================================")

# ===============================================================================================
# Task 3: Construct a basic Transformer model
# ===============================================================================================
# Construct a basic transformer model with appropriate layers (Embedding, TransformerEncoder, GlobalAveragePooling1D, Dense)
print ("==============================================================================")
print("Task 3: Constructing a basic Transformer model...")
print ("==============================================================================")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer
inputs = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim=max_words, output_dim=embed_dim)(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)(embedding_layer)
pooling_layer = layers.GlobalAveragePooling1D()(transformer_block)
dropout_layer = layers.Dropout(0.1)(pooling_layer)
outputs = layers.Dense(1, activation="sigmoid")(dropout_layer)
basic_transformer_model = keras.Model(inputs=inputs, outputs=outputs)
basic_transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
basic_transformer_model.summary()

# ===============================================================================================
# Task 4: Train the basic model
# ===============================================================================================
# Train the model on the IMDB training data for 5 epochs using binary_crossentropy
print ("==============================================================================")
print("Task 4: Training the basic Transformer model...")
print ("==============================================================================")
history_basic = basic_transformer_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))
print ("==============================================================================")
print("Task 4: Basic Transformer model training complete.")
print ("==============================================================================")

# ===============================================================================================
# Task 5: Display model architecture and training progress
# ===============================================================================================
# Display model architecture and plot training performance (accuracy, loss)
print ("==============================================================================")
print("Task 5: Displaying model architecture and training performance...")
print ("==============================================================================")
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_basic.history['accuracy'])
plt.plot(history_basic.history['val_accuracy'])
plt.title('Basic Transformer Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_basic.history['loss'])
plt.plot(history_basic.history['val_loss'])
plt.title('Basic Transformer Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
print ("==============================================================================")
print("Task 5: Model architecture and training performance displayed.")
print ("==============================================================================")

# ===============================================================================================
# Task 6: Construct an advanced transformer model
# ===============================================================================================
#  Implement a custom positional encoding layer &  Use a Transformer block with multi-head attention (num_heads ≥ 4) and Dropout
print ("==============================================================================")
print("Task 6: Constructing an advanced Transformer model...")
print ("==============================================================================")
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding,
        })
        return config

    def positional_encoding(self, max_len, embed_dim):
        angle_rads = self.get_angles(np.arange(max_len)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
embed_dim_adv = 64  # Embedding size for each token
num_heads_adv = 4  # Number of attention heads
ff_dim_adv = 128  # Hidden layer size in feed forward network inside transformer
inputs_adv = layers.Input(shape=(max_len,))
embedding_layer_adv = layers.Embedding(input_dim=max_words, output_dim=embed_dim_adv)(inputs_adv)
pos_encoding_layer = PositionalEncoding(max_len, embed_dim_adv)(embedding_layer_adv)
transformer_block_adv = TransformerBlock(embed_dim_adv, num_heads_adv, ff_dim_adv, rate=0.2)(pos_encoding_layer)
pooling_layer_adv = layers.GlobalAveragePooling1D()(transformer_block_adv)
dropout_layer_adv = layers.Dropout(0.2)(pooling_layer_adv)
outputs_adv = layers.Dense(1, activation="sigmoid")(dropout_layer_adv)
advanced_transformer_model = keras.Model(inputs=inputs_adv, outputs=outputs_adv)
advanced_transformer_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
advanced_transformer_model.summary()

# ===============================================================================================
# Task 7: Train the advanced model
# ===============================================================================================
# Train the enhanced model and display the training curves and model summary
print ("==============================================================================")
print("Task 7: Training the advanced Transformer model...")
print ("==============================================================================")
history_advanced = advanced_transformer_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))
# Display model summary
advanced_transformer_model.summary()

print ("==============================================================================")    
print("Task 7: Advanced Transformer model training complete.")
print ("==============================================================================")

# ===============================================================================================
# Task 8: Display model architecture and training progress
# ===============================================================================================
# Display model architecture and plot training performance (accuracy, loss)
print ("==============================================================================")
print("Task 8: Displaying advanced model architecture and training performance...")
print ("==============================================================================")
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_advanced.history['accuracy'])
plt.plot(history_advanced.history['val_accuracy'])
plt.title('Advanced Transformer Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history_advanced.history['loss'])
plt.plot(history_advanced.history['val_loss'])
plt.title('Advanced Transformer Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
print ("==============================================================================")
print("Task 8: Advanced model architecture and training performance displayed.")
print ("==============================================================================")

# ===============================================================================================
# Task 9: Compare training and validation results for each model
# ===============================================================================================
# Evaluate both models using Accuracy, Precision, Recall, F1-score and AUC-ROC
print ("==============================================================================")
print("Task 9: Comparing training and validation results for both models...")
print ("==============================================================================")
# Evaluate basic model
y_val_pred_basic = (basic_transformer_model.predict(X_val) > 0.5).astype("int32")
print("Basic Transformer Model Classification Report:")
print(classification_report(y_val, y_val_pred_basic))
# Evaluate advanced model
y_val_pred_advanced = (advanced_transformer_model.predict(X_val) > 0.5).astype("int32")
print("Advanced Transformer Model Classification Report:")
print(classification_report(y_val, y_val_pred_advanced))
# Plot confusion matrices
conf_matrix_basic = confusion_matrix(y_val, y_val_pred_basic)
conf_matrix_advanced = confusion_matrix(y_val, y_val_pred_advanced)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_basic, annot=True, fmt='d', cmap='Blues')
plt.title('Basic Transformer Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_advanced, annot=True, fmt='d', cmap='Greens')
plt.title('Advanced Transformer Model Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print ("==============================================================================")
print("Task 9: Comparison complete.")
print ("==============================================================================")
