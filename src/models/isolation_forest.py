import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load the newly generated features
df = pd.read_csv("data/features.csv")

# Define the feature set for the model
feature_cols = [
    "session_duration",
    "n_events",
    "click_rate",
    "avg_dwell_time",
    "click_variance",
    "page_entropy"
]
X = df[feature_cols]

# Initialize and train the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

# Save the trained model
joblib.dump(model, "artifacts/models/iforest.pkl")
print("Saved -> artifacts/models/iforest.pkl")
