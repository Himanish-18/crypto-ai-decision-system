import hashlib
import json
import time
from pathlib import Path
from typing import Dict


class ImmutableAuditLog:
    """
    Audit log where each entry contains simple SHA-256 hash of previous entry.
    """

    def __init__(self, log_path: str = "logs/compliance/audit_chain.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        if not self.log_path.exists():
            return "0" * 64

        # Read last line
        with open(self.log_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return "0" * 64
            last_entry = json.loads(lines[-1])
            return last_entry.get("current_hash", "0" * 64)

    def log_event(self, event_type: str, details: Dict):
        """
        Append event to chain.
        """
        timestamp = time.time()
        payload = {
            "timestamp": timestamp,
            "type": event_type,
            "details": details,
            "prev_hash": self.last_hash,
        }

        # Calculate Hash
        payload_str = json.dumps(payload, sort_keys=True)
        current_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        entry = payload
        entry["current_hash"] = current_hash

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.last_hash = current_hash
