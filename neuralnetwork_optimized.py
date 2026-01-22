import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, TextVectorization
import numpy as np
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print("Found GPU:", gpu)
else:
    print("No GPU found for TensorFlow. Using CPU.")

# 1. PREPARE DATA

file_path = 'text.txt'

# Check if file exists to avoid crash
if not os.path.exists(file_path):
    print(f"Error: '{file_path}' not found. Please create it first.")
    exit()

with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Cleaning text
text = text.lower().replace("â€”", " ").replace('\n', '').replace('\r', '')


# 2. VECTORIZATION

vectorize_layer = TextVectorization(
    max_tokens=None,
    standardize="lower_and_strip_punctuation",
    split="character",
    output_mode="int",
    output_sequence_length=None
)

# Adapt the layer to learn the vocabulary
vectorize_layer.adapt([text])

# 1. Convert text to tensor
# We keep it as a Tensor, not a Numpy array, for efficiency
vectorized_text = vectorize_layer(text)

# 2. Create a Dataset from the tensor
# This handles the sliding window automatically
sequence_length = 100

# Create sequences of length 101 (100 input + 1 target)
ds = tf.data.Dataset.from_tensor_slices(vectorized_text)
ds = ds.window(sequence_length + 1, shift=1, drop_remainder=True)

# Flatten the windows into individual sequences
ds = ds.flat_map(lambda window: window.batch(sequence_length + 1))

# Split into Input (X) and Target (y)
# X = first 40 chars, y = last char
ds = ds.map(lambda window: (window[:-1], window[-1]))

# 3. Batch and Prefetch
# Batch size 64
ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)

print("Dataset prepared.")

# 4. BUILD THE MODEL

vocab_size = len(vectorize_layer.get_vocabulary())
embedding_dim = 32
lstm_units = 128

model = Sequential()

# Layer 1: Embedding 
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

# Layer 2: LSTM 
model.add(LSTM(units=lstm_units, return_sequences=True))

# Layer 3: Dense 
model.add(Dense(units=vocab_size, activation='softmax'))


# 5. COMPILE AND TRAIN

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Starting Training...")
# Use a low number of epochs (e.g., 5) for testing, increase (e.g., 30-50) for better results
model.fit(ds, epochs=5)


# 6. SAVE MODEL AND VOCAB

# Save the Keras model
model.save('my_character_generator.keras')
print("Model saved to 'my_character_generator.keras'")

# Save the vocabulary list manually so we can decode later
vocab_list = vectorize_layer.get_vocabulary()
with open('vocab.txt', 'w', encoding='utf-8') as f:
    for char in vocab_list:
        f.write(char + "\n")
print("Vocabulary saved to 'vocab.txt'")


# 7. GENERATE TEXT

print("\nGenerating Text...")

# Seed text
seed = "the weather was really good and i really"

# Vectorize seed
vec_seed = vectorize_layer(seed).numpy()

# Ensure it's exactly 100 long and reshape for (Batch, TimeSteps)
vec_seed = vec_seed[:100].reshape(1, 100)

# Temperature settings (0.7 is a good balance, 0 is neutral, 1 is very random)
temperature = 0.7
generated_indices = []

# Loop to generate 100 characters
for i in range(100):
    prediction = model.predict(vec_seed, verbose=0)
    
    # Get the probability list
    preds = prediction[0]
    
    # Apply Temperature logic (make the distribution smoother or sharper)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    # Sample from the probabilities
    next_index = np.random.choice(len(preds), p=preds)
    generated_indices.append(next_index)
    
    # Update Seed: Drop first char, append new char
    vec_seed = vec_seed[:, 1:]
    vec_seed = np.append(vec_seed, [[next_index]], axis=1)

# Decode Indices back to Text
predicted_chars = [vocab_list[i] for i in generated_indices]
final_text = "".join(predicted_chars)

print("Generated Output:")
print(final_text)
