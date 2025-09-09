import pandas as pd
import joblib
import numpy as np
from tensorflow import keras
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Parameters ---
MAX_SEQ_LENGTH = 15  # Must match the value in lstm_autoencoder.py

def load_sequences():
    """Loads all sequences and their corresponding session IDs."""
    sequences = []
    session_ids = []
    with open("data/sequences.jsonl") as f:
        for line in f:
            data = json.loads(line)
            sequences.append(data["sequence"])
            session_ids.append(data["session_id"])
    return sequences, session_ids

def main():
    # --- Load Data and Models ---
    df = pd.read_csv("data/features.csv")
    feature_cols = [
        "session_duration", "n_events", "click_rate", 
        "avg_dwell_time", "click_variance", "page_entropy"
    ]
    tabular_X = df[feature_cols].values

    iforest = joblib.load("artifacts/models/iforest.pkl")
    lstm_ae = keras.models.load_model("artifacts/models/lstm_ae.keras", compile=False)

    # --- 1. Calculate Isolation Forest Scores ---
    # The decision_function gives a score where lower is more anomalous. We invert it.
    if_scores = -iforest.decision_function(tabular_X)
    # Normalize scores to be between 0 and 1
    df["iforest_score"] = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())

    # --- 2. Calculate LSTM Autoencoder Scores ---
    sequences, session_ids = load_sequences()
    X_padded = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
    X_padded = X_padded.reshape((X_padded.shape[0], X_padded.shape[1], 1))
    
    # Get the model's reconstruction of the input sequences
    X_pred = lstm_ae.predict(X_padded)
    
    # Calculate reconstruction error (MSE) for each sequence
    mse = np.mean(np.power(X_padded - X_pred, 2), axis=(1, 2))
    
    # Create a DataFrame for LSTM scores to merge them correctly
    lstm_scores_df = pd.DataFrame({'session_id': session_ids, 'lstm_score_raw': mse})
    
    # Normalize the MSE scores to be between 0 and 1
    min_mse = lstm_scores_df['lstm_score_raw'].min()
    max_mse = lstm_scores_df['lstm_score_raw'].max()
    # Handle case where all errors are the same to avoid division by zero
    if max_mse > min_mse:
        lstm_scores_df["lstm_score"] = (lstm_scores_df['lstm_score_raw'] - min_mse) / (max_mse - min_mse)
    else:
        lstm_scores_df["lstm_score"] = 0.0

    # Merge LSTM scores into the main DataFrame using session_id
    df = pd.merge(df, lstm_scores_df[['session_id', 'lstm_score']], on='session_id')

    # --- 3. Fuse Scores into a Hybrid Score ---
    df["hybrid_score"] = 0.4 * df["iforest_score"] + 0.6 * df["lstm_score"]
    
    # (This is your placeholder logic for ground truth labels)
    df["label"] = df["session_id"].apply(lambda x: 1 if "bot" in str(x).lower() else 0)

    # --- Save Final Scores ---
    df.to_csv("artifacts/scores.csv", index=False)
    print("Saved -> artifacts/scores.csv")

if __name__ == "__main__":
    main()
