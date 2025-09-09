import pandas as pd
import numpy as np
import json
from scipy.stats import entropy
from collections import Counter
import os

def calculate_features(session_events, page_map):
    """Calculates all features for a single session."""
    features = {}

    n_events = len(session_events)
    if n_events == 0:
        # Return a dictionary with zero values for all features if no events
        return {
            "session_duration": 0,
            "n_events": 0,
            "click_rate": 0,
            "avg_dwell_time": 0,
            "click_variance": 0,
            "page_entropy": 0,
            "click_sequence": []
        }

    # Total time of the session in milliseconds
    session_duration = session_events[-1]["t_rel_ms"]
    
    # Calculate time between each click (dwell time)
    timestamps = [0] + [e["t_rel_ms"] for e in session_events]
    dwell_times = np.diff(timestamps)

    # --- Feature Calculations ---
    features["session_duration"] = session_duration
    features["n_events"] = n_events
    # Click rate: number of clicks per second
    features["click_rate"] = n_events / (session_duration / 1000) if session_duration > 0 else 0
    features["avg_dwell_time"] = np.mean(dwell_times)
    features["click_variance"] = np.var(dwell_times) if n_events > 1 else 0

    # Page Entropy: randomness of page visits
    pages = [e["page"] for e in session_events]
    page_counts = Counter(pages)
    page_probabilities = [count / n_events for count in page_counts.values()]
    features["page_entropy"] = entropy(page_probabilities, base=2)

    # Encoded Click Sequence for LSTM model
    features["click_sequence"] = [page_map[p] for p in pages]
    
    return features


def main():
    """
    Main function to read raw session logs, generate features,
    and save them into separate files for tabular and sequence models.
    """
    sessions_data = []
    with open("logs/sessions.jsonl") as f:
        for line in f:
            sessions_data.append(json.loads(line))
    
    # Create a consistent mapping from page names to integers
    all_pages = set()
    for s in sessions_data:
        for event in s["events"]:
            all_pages.add(event["page"])
    page_map = {page: i for i, page in enumerate(sorted(list(all_pages)))}
    
    # Ensure artifacts/models directory exists
    os.makedirs("artifacts/models", exist_ok=True)
    
    # Save the page map for the LSTM model to use later
    with open("artifacts/models/page_map.json", "w") as f:
        json.dump(page_map, f)
    print("Saved -> artifacts/models/page_map.json")

    # Calculate features for every session
    all_features = []
    for session in sessions_data:
        session_id = session["session_id"]
        events = session["events"]
        
        features = calculate_features(events, page_map)
        features["session_id"] = session_id
        all_features.append(features)

    # Separate tabular features from the click sequences
    tabular_features = []
    sequences = {}
    for f in all_features:
        sequences[f["session_id"]] = f.pop("click_sequence")
        tabular_features.append(f)
        
    # Create and save the tabular features to a CSV file
    df_features = pd.DataFrame(tabular_features)
    cols = ['session_id', 'session_duration', 'n_events', 'click_rate', 'avg_dwell_time', 'click_variance', 'page_entropy']
    df_features = df_features[cols]
    df_features.to_csv("data/features.csv", index=False)
    print("Saved -> data/features.csv")

    # Save the encoded sequences to a JSONL file
    with open("data/sequences.jsonl", "w") as f:
        for session_id, seq in sequences.items():
            f.write(json.dumps({"session_id": session_id, "sequence": seq}) + "\n")
    print("Saved -> data/sequences.jsonl")


if __name__ == "__main__":
    main()
