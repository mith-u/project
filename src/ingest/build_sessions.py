import json, pandas as pd

sessions = []
with open("logs/sessions.jsonl") as f:
    for line in f:
        s = json.loads(line)
        duration = s["events"][-1]["t_rel_ms"] if s["events"] else 0
        sessions.append({"session_id": s["session_id"], "duration": duration, "n_events": len(s["events"])})

df = pd.DataFrame(sessions)
df.to_csv("data/raw_sessions.csv", index=False)
print("Saved -> data/raw_sessions.csv")
