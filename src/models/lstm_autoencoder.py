import numpy as np
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Parameters ---
MAX_SEQ_LENGTH = 15  # Max length of a session sequence
EMBEDDING_DIM = 8   # Dimension for page embeddings

def load_data():
    """Loads sequences from the jsonl file."""
    sequences = []
    with open("data/sequences.jsonl") as f:
        for line in f:
            sequences.append(json.loads(line)["sequence"])
    return sequences

def main():
    """Trains the LSTM Autoencoder on the clickstream sequences."""
    sequences = load_data()

    # Pad sequences to ensure they all have the same length
    X_padded = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')

    # Reshape data for LSTM input: (samples, timesteps, features)
    X_padded = X_padded.reshape((X_padded.shape[0], X_padded.shape[1], 1))

    # --- Define the Autoencoder Model Architecture ---
    model = keras.Sequential([
        # Encoder part
        layers.LSTM(32, activation="relu", input_shape=(MAX_SEQ_LENGTH, 1), return_sequences=True),
        layers.LSTM(16, activation="relu", return_sequences=False),
        layers.RepeatVector(MAX_SEQ_LENGTH),
        # Decoder part
        layers.LSTM(16, activation="relu", return_sequences=True),
        layers.LSTM(32, activation="relu", return_sequences=True),
        layers.TimeDistributed(layers.Dense(1))
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # --- Train the Model ---
    # The model learns to reconstruct its own input.
    # Anomalies will have a higher reconstruction error.
    model.fit(X_padded, X_padded, epochs=20, batch_size=16, shuffle=True, verbose=2)

    # --- Save the Trained Model ---
    model.save("artifacts/models/lstm_ae.keras")
    print("\nSaved -> artifacts/models/lstm_ae.keras")

if __name__ == "__main__":
    main()
