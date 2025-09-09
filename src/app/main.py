import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os, json
import plotly.express as px

st.title("🔍 Web Anomaly Detection Dashboard")

# Check if scores file exists
if not os.path.exists("artifacts/scores.csv"):
    st.warning("⚠️ No scores found. Please run fuse.py to generate artifacts/scores.csv.")
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

        # 🔎 Timeline visualization (using raw logs)
        if os.path.exists("logs/sessions.jsonl"):
            events = []
            with open("logs/sessions.jsonl") as f:
                for line in f:
                    s = json.loads(line)
                    if str(s["session_id"]) == str(session_id):
                        events = s["events"]
                        break

            if events:
                ev_df = pd.DataFrame(events)
                fig2 = px.line(ev_df, x="t_rel_ms", y="page", markers=True,
                               title=f"Session {session_id} Timeline",
                               labels={"t_rel_ms":"Time (ms)", "page":"Visited Page"})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No event data found for this session.")
    # 🔎 Side-by-Side Session Comparison
    st.subheader("Compare Normal vs Anomalous Session Timelines")

    if os.path.exists("logs/sessions.jsonl"):

        # Split sessions into normal & anomalous based on threshold
        normal_sessions = df[df["hybrid_score"] < threshold]["session_id"].unique()
        anomalous_sessions = df[df["hybrid_score"] >= threshold]["session_id"].unique()

        if len(normal_sessions) > 0 and len(anomalous_sessions) > 0:
            col1, col2 = st.columns(2)

            with col1:
                normal_choice = st.selectbox("Select Normal Session", normal_sessions, key="norm")
            with col2:
                anom_choice = st.selectbox("Select Anomalous Session", anomalous_sessions, key="anom")

            def load_events(session_id):
                with open("logs/sessions.jsonl") as f:
                    for line in f:
                        s = json.loads(line)
                        if str(s["session_id"]) == str(session_id):
                            return pd.DataFrame(s["events"])
                return pd.DataFrame()

            norm_ev = load_events(normal_choice)
            anom_ev = load_events(anom_choice)

            if not norm_ev.empty and not anom_ev.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### Normal Session {normal_choice}")
                    fig_norm = px.line(norm_ev, x="t_rel_ms", y="page", markers=True,
                                       title="Navigation Timeline",
                                       labels={"t_rel_ms":"Time (ms)", "page":"Page"})
                    st.plotly_chart(fig_norm, use_container_width=True)

                with col2:
                    st.markdown(f"### Anomalous Session {anom_choice}")
                    fig_anom = px.line(anom_ev, x="t_rel_ms", y="page", markers=True,
                                       title="Navigation Timeline",
                                       labels={"t_rel_ms":"Time (ms)", "page":"Page"})
                    st.plotly_chart(fig_anom, use_container_width=True)
            else:
                st.info("Could not load events for selected sessions.")
        else:
            st.warning("Not enough normal or anomalous sessions for comparison.")
