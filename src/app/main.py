import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.title("🔍 Web Anomaly Detection Dashboard")
st.write("MITHUN 22BCE2307")

# Check if scores file exists
if not os.path.exists("artifacts/scores.csv"):
    st.warning("⚠️ No scores found. Please run the pipeline first (fuse.py) to generate artifacts/scores.csv.")
else:
    df = pd.read_csv("artifacts/scores.csv")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Hybrid Anomaly Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["hybrid_score"], bins=30, edgecolor="black")
    ax.set_xlabel("Hybrid Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Anomaly Threshold")
    threshold = st.slider("Set threshold", 0.0, 1.0, 0.7)
    anomalies = df[df["hybrid_score"] >= threshold]

    st.write(f"Detected {len(anomalies)} anomalies above threshold {threshold}")
    st.dataframe(anomalies.sort_values("hybrid_score", ascending=False).head(20))

    st.subheader("Inspect Session")
    if not anomalies.empty:
        session_id = st.selectbox("Select Session ID", anomalies["session_id"].unique())
        session_data = df[df["session_id"] == session_id].iloc[0]
        st.json(session_data.to_dict())
