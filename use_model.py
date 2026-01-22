import tensorflow as tf
import numpy as np
import sys

print("Loading model...")
try:
    model = tf.keras.models.load_model('my_character_generator.keras')
except OSError:
    print("Error: 'my_character_generator.keras' not found in this folder.")
    sys.exit()

print("Loading vocabulary...")
try:
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        # rstrip('\n') removes ONLY the newline, keeping the space character ' ' 
        vocab_list = [line.rstrip('\n') for line in f]
    
    # Remove accidental duplicates or empty lines (but keep the actual space ' ')
    # dict.fromkeys preserves order
    vocab_list = list(dict.fromkeys(vocab_list))

except FileNotFoundError:
    print("Error: 'vocab.txt' not found in this folder.")
    sys.exit()

# Recreate the TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=None, 
    standardize=None,
    split="character", 
    output_mode="int",
    vocabulary=vocab_list
)

# 2. GENERATION FUNCTION

def generate_paragraph(seed_text, chars_to_generate=300, temperature=0.4):
    """
    Takes a seed string, generates N characters using the model.
    """
    # Vectorize the seed
    vec_seed = vectorize_layer(seed_text).numpy()
    sequence_length = 40 
    
    # If the user typed less than 40 characters:
    if len(vec_seed) < sequence_length:

        missing = sequence_length - len(vec_seed)
        vec_seed = np.concatenate([vec_seed, vec_seed[:missing]])
    
    elif len(vec_seed) > sequence_length:
        vec_seed = vec_seed[:sequence_length]

    # Reshape for the model: (Batch_Size, Sequence_Length)
    vec_seed = vec_seed.reshape(1, sequence_length)

    generated_indices = []

    print(f"Generating {chars_to_generate} characters...")
    
    for i in range(chars_to_generate):
        # Predict
        prediction = model.predict(vec_seed, verbose=0)
        
        # Get probabilities
        preds = prediction[0]
        
        # Apply Temperature Sampling
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # Choose next index
        next_index = np.random.choice(len(preds), p=preds)

        generated_indices.append(next_index)
        
        # Update the sliding window
        vec_seed = vec_seed[:, 1:]
        vec_seed = np.append(vec_seed, [[next_index]], axis=1)

    # Decode indices back to characters
    generated_chars = [vocab_list[i] for i in generated_indices]
    return "".join(generated_chars)


print("\n--- Text Generator Ready ---")
print("Type a sentence to start the AI.")
print("Type 'quit' to exit.\n")

while True:
    user_input = input("Enter a starting sentence: ")
    
    if user_input.lower() == 'quit':
        break
    
    if not user_input.strip():
        continue

    # Generate the paragraph
    result = generate_paragraph(user_input, chars_to_generate=300, temperature=0.7)
    
    print("\n--- Generated Paragraph ---\n")
    print(result)
    print("-" * 40)