from faker import Faker
import random, time, json

fake = Faker()

def generate_session(session_id):
    pages = ["home", "products", "cart", "checkout"]
    events = []
    t = 0
    for _ in range(random.randint(3, 6)):
        page = random.choice(pages)
        t += random.randint(500, 3000)  # milliseconds
        events.append({"page": page, "t_rel_ms": t})
    return {"session_id": session_id, "events": events}

if __name__ == "__main__":
    sessions = [generate_session(i) for i in range(5)]
    with open("logs/sessions.jsonl", "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
