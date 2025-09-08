import pandas as pd, joblib, numpy as np
from tensorflow import keras

df = pd.read_csv("data/features.csv")
X = df[["duration","n_events","avg_gap","burstiness"]]

iforest = joblib.load("artifacts/models/iforest.pkl")
lstm = keras.models.load_model("artifacts/models/lstm_ae.keras", compile=False)




if_scores = -iforest.decision_function(X)  # anomaly score
lstm_scores = np.random.rand(len(X))       # placeholder for reconstruction error

df["iforest_score"] = (if_scores - if_scores.min()) / (if_scores.max()-if_scores.min())
df["lstm_score"] = (lstm_scores - lstm_scores.min()) / (lstm_scores.max()-lstm_scores.min())
df["hybrid_score"] = 0.4*df["iforest_score"] + 0.6*df["lstm_score"]

df.to_csv("artifacts/scores.csv", index=False)
print("Saved -> artifacts/scores.csv")
