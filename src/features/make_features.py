import pandas as pd, numpy as np

df = pd.read_csv("data/raw_sessions.csv")
df["avg_gap"] = df["duration"] / df["n_events"].replace(0,1)
df["burstiness"] = np.random.rand(len(df))  # placeholder for real calc
df.to_csv("data/features.csv", index=False)
print("Saved -> data/features.csv")
