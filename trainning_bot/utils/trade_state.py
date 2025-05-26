import os
import json

STATE_PATH = "trade/state.json"

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return {"entry_price": 0, "holding": False}

def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f)
