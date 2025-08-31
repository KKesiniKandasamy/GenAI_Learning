#  Train, and evaluate two neural network architectures (basic and advanced Transformer models), and compare their performance on sentiment classification using appropriate metrics.
# Task 1: Utilise Libraries/Dataset
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

# Task 2: Data Processing and Exploration
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

# Task 3: Construct a basic Transformer model
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
