from faker import Faker
import random
import time
import json

fake = Faker()

def generate_session(session_id, is_bot=False):
    """Generates a session with events. Bots have more predictable, rapid behavior."""
    pages = ["home", "products", "cart", "checkout"]
    events = []
    t = 0
    
    if is_bot:
        # Bot behavior: fewer clicks, faster, more linear path
        num_events = random.randint(2, 4)
        path = random.sample(pages, num_events) # Less random path
        for page in path:
            t += random.randint(200, 1000)  # Bots are faster
            events.append({"page": page, "t_rel_ms": t})
    else:
        # Human behavior: more clicks, more random timing and path
        for _ in range(random.randint(4, 8)): # Increased human activity slightly
            page = random.choice(pages)
            t += random.randint(1500, 5000)  # Humans are slower
            events.append({"page": page, "t_rel_ms": t})
            
    return {"session_id": session_id, "events": events}

if __name__ == "__main__":
    sessions = []
    # --- Generate a much larger dataset ---
    num_human_sessions = 1000
    num_bot_sessions = 150

    # Generate human sessions with numeric IDs
    for i in range(num_human_sessions):
        sessions.append(generate_session(i, is_bot=False))

    # Generate bot sessions with "bot" in the ID
    for i in range(num_bot_sessions):
        session_id = f"bot_{i + num_human_sessions}"
        sessions.append(generate_session(session_id, is_bot=True))
        
    random.shuffle(sessions) # Mix them up

    with open("logs/sessions.jsonl", "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    
    print(f"Generated {len(sessions)} total sessions ({num_human_sessions} human, {num_bot_sessions} bot).")
    print("Saved -> logs/sessions.jsonl")

