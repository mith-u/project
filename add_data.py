import pandas as pd
import numpy as np

# --- Configuration ---
# The path is now relative to where you are running the script from
file_path = "artifacts/scores.csv"
num_human_samples = 5  # The number of "human" samples (label=0) to add

# --- Load Your Data ---
try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded your file: {file_path}")
except FileNotFoundError:
    print(f"❌ Error: Could not find the file at the path: {file_path}")
    print("Please make sure you are running this script from your project's main directory (`C:\\Users\\block\\Downloads\\project-main>`).")
    exit()

# --- Generate New "Human" Data ---
new_rows = []
for i in range(num_human_samples):
    template_row = df.iloc[np.random.randint(0, len(df))].copy()
    
    template_row['session_id'] = f'human_session_{i+1}'
    template_row['label'] = 0
    
    template_row['session_duration'] *= np.random.uniform(0.9, 1.2)
    template_row['n_events'] += np.random.randint(1, 5)
    template_row['iforest_score'] *= np.random.uniform(0.5, 0.8)
    template_row['lstm_score'] *= np.random.uniform(0.4, 0.7)
    template_row['hybrid_score'] *= np.random.uniform(0.6, 0.9)
    
    new_rows.append(template_row)

# --- Combine and Save the Updated Data ---
new_df = pd.DataFrame(new_rows)
updated_df = pd.concat([df, new_df], ignore_index=True)
updated_df.to_csv(file_path, index=False)

print(f"scores.csv file created.")
