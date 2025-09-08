import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

df = pd.read_csv("data/features.csv")
X = df[["duration","n_events","avg_gap","burstiness"]]

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

joblib.dump(model, "artifacts/models/iforest.pkl")
print("Saved -> artifacts/models/iforest.pkl")
