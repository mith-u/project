import pandas as pd
import numpy as np
import os

def create_synthetic_labels(features_path="data/features.csv", output_path="labels.csv"):
    """
    Generates a synthetic labels.csv file based on heuristic rules
    to simulate a ground truth for anomaly detection.
    """
    if not os.path.exists(features_path):
        print(f"❌ Error: Features file not found at '{features_path}'")
        print("Please run 'python src/features/make_features.py' first.")
        return

    df = pd.read_csv(features_path)
    print(f"✅ Found features file with {len(df)} sessions.")

    # Initialize all labels to 0 (normal)
    df['label'] = 0

    # --- Define Heuristic Rules for Anomalies ---
    # Rule 1: Very high click rate (potential bot activity)
    high_click_rate_threshold = df['click_rate'].quantile(0.95)
    df.loc[df['click_rate'] > high_click_rate_threshold, 'label'] = 1

    # Rule 2: Short session with many events (potential scraper)
    low_duration_threshold = df['session_duration'].quantile(0.10)
    high_events_threshold = df['n_events'].quantile(0.80)
    df.loc[(df['session_duration'] < low_duration_threshold) & (df['n_events'] > high_events_threshold), 'label'] = 1
    
    # Rule 3: Very low page entropy (repetitive, non-human behavior)
    low_entropy_threshold = df['page_entropy'].quantile(0.05)
    df.loc[df['page_entropy'] < low_entropy_threshold, 'label'] = 1
    
    # --- Save the labels to a CSV file in the root directory ---
    output_df = df[['session_id', 'label']]
    output_df.to_csv(output_path, index=False)

    n_anomalies = output_df['label'].sum()
    total_sessions = len(output_df)
    anomaly_percentage = (n_anomalies / total_sessions) * 100

    print(f"✅ Successfully created '{output_path}'.")
    print(f"   - Total sessions processed: {total_sessions}")
    print(f"   - Identified {n_anomalies} anomalies ({anomaly_percentage:.2f}%) based on rules.")

if __name__ == "__main__":
    create_synthetic_labels()
