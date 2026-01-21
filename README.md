# Character-Prediction-Neural-Network
This is a simple and begineer friendly neural network I built while learning about keras and tensorflow. This version of the code is a non optimized and slow on cpu, which i am looking forward to further edits. 
This project uses a recurrent neural network (LSTM) to learn the structure of language from a text file and generate new text character by character. 

Model Architecture 

The model is built using the Keras Sequential API and consists of three layers. 

    Embedding Layer 
         Output Dimension: 32
         Function: This layer converts integer inputs (representing characters) into dense vectors of fixed size. It learns a representation for every character in the dataset.
          

    LSTM Layer (Long Short-Term Memory) 
         Units: 128
         Function: This layer acts as the brain of the network. It processes the sequence of 100 characters step by step, maintaining an internal memory state that allows it to remember context from earlier in the sequence to predict the next character.
          

    Dense Layer 
         Units: Variable (Equal to the vocabulary size)
         Activation: Softmax
         Function: This is the output layer. It receives the final state from the LSTM and outputs a probability distribution across all possible characters, indicating the likelihood of each one being the next character.
          

Process Overview 

The operation of this network is divided into two main phases: Training and Generation. 
1. Training Phase 

The goal of training is to teach the model which character typically follows another. 

     Text Cleaning: The raw text file is converted to lowercase and stripped of punctuation and newlines to standardize the input.
     Vectorization: A TextVectorization layer maps every unique character in the text to a unique integer ID.
     Dataset Creation: The text is sliced into sliding windows of 100 characters.
         Input (X): A sequence of 100 characters (e.g., characters 0-99).
         Target (y): The immediate next character (e.g., character 100).
         
     Learning: The model takes a sequence of 100 integers, passes them through the embedding and LSTM layers, and attempts to predict the next integer. The loss is calculated using sparse_categorical_crossentropy, and the weights of the network are adjusted to reduce prediction errors.
     

2. Generation Phase 

Once trained, the model can be used to create new text. 

     Seed Input: A starting string is provided to the model (must be converted to integers and padded to length 100).
     Prediction: The model predicts the probabilities for the next character.
     Temperature Sampling: To prevent repetitive output, the probabilities are adjusted by a temperature value before sampling. This allows the model to pick slightly less probable characters occasionally, adding creativity to the text.
     Sliding Window: The chosen character is appended to the end of the input sequence, and the first character is removed. This new sequence is fed back into the model. This process repeats for the desired length of the generated text.
     
