
import json
import time
from pathlib import Path

class TransactionReplayLog:
    """
    Logs every state change for full system replay capability.
    Format: JSONL (Time, Component, Action, StateDelta)
    """
    
    def __init__(self, log_path: str = "logs/replay_v26.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def log_action(self, component: str, action_type: str, state_delta: dict):
        entry = {
            "ts": time.time_ns(),
            "component": component,
            "action": action_type,
            "delta": state_delta
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
