import os
from collections import deque
from src.guardian.monitors.base_monitor import BaseMonitor, Anomaly
from datetime import datetime

class LogicMonitor(BaseMonitor):
    """
    Validates business logic health.
    Primary Check: 'Neutral Lock' where prediction stays Neutral > X times.
    """
    def __init__(self, pred_log_path="predictions.log", lock_limit=10):
        super().__init__(name="LogicMonitor")
        self.pred_log = pred_log_path
        self.lock_limit = lock_limit
        self.last_pos = 0
        self.neutral_streak = 0

        # Init pos
        if os.path.exists(self.pred_log):
            self.last_pos = os.path.getsize(self.pred_log)

    def check(self):
        anomalies = []
        if not os.path.exists(self.pred_log):
            return []

        try:
            curr_size = os.path.getsize(self.pred_log)
            if curr_size < self.last_pos:
                self.last_pos = 0

            if curr_size == self.last_pos:
                return []

            with open(self.pred_log, 'r') as f:
                f.seek(self.last_pos)
                lines = f.readlines()
                self.last_pos = curr_size

            for line in lines:
                parts = line.split(',')
                # Format: Timestamp, Direction, Price...
                if len(parts) > 2:
                    direction = parts[1]
                    if direction == "Neutral":
                        self.neutral_streak += 1
                    else:
                        self.neutral_streak = 0
            
            if self.neutral_streak >= self.lock_limit:
                anomalies.append(Anomaly(
                    type="NEUTRAL_LOCK",
                    severity="CRITICAL",
                    details={"streak": self.neutral_streak, "limit": self.lock_limit},
                    timestamp=datetime.now()
                ))
                # Reset streak to avoid spamming the same event every cycle?
                # Or keep it high until healed? 
                # Let's keep reporting until fixed.

        except Exception as e:
            self.logger.error(f"Logic Validation Failed: {e}")

        return anomalies
