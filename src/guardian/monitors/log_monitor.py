import re
import os
from datetime import datetime
from collections import deque
from src.guardian.monitors.base_monitor import BaseMonitor, Anomaly

class LogMonitor(BaseMonitor):
    """
    Tails a log file and searches for regex patterns (Errors, Tracebacks).
    """
    def __init__(self, filepath: str, max_lines_overlap=50):
        super().__init__(name=f"LogMonitor::{os.path.basename(filepath)}")
        self.filepath = filepath
        self.last_pos = 0
        self.patterns = {
            r"ERROR": "LOG_ERROR",
            r"CRITICAL": "LOG_CRITICAL",
            r"Traceback": "TRACEBACK"
        }
        
        # Init file position to end to avoid reading old logs on startup
        if os.path.exists(filepath):
            self.last_pos = os.path.getsize(filepath)

    def check(self):
        anomalies = []
        if not os.path.exists(self.filepath):
            return []

        try:
            curr_size = os.path.getsize(self.filepath)
            if curr_size < self.last_pos:
                self.last_pos = 0  # File rotated

            if curr_size == self.last_pos:
                return []

            with open(self.filepath, 'r') as f:
                f.seek(self.last_pos)
                new_data = f.read()
                self.last_pos = curr_size

            for line in new_data.split('\n'):
                for pattern, anomaly_type in self.patterns.items():
                    if re.search(pattern, line):
                        anomalies.append(Anomaly(
                            type=anomaly_type,
                            severity="CRITICAL" if "CRITICAL" in anomaly_type else "WARNING",
                            details={"file": self.filepath, "match": line.strip()},
                            timestamp=datetime.now()
                        ))
        except Exception as e:
            self.logger.error(f"Failed to read log: {e}")

        return anomalies
