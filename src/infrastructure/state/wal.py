
import time
import json
import os
import logging

# v40 Infrastructure: Write-Ahead Logging (WAL)
# Ensures durability by logging transactions BEFORE applying them.
# Supports replay for crash recovery.

class WriteAheadLog:
    def __init__(self, log_path="logs/wal.log"):
        self.log_path = log_path
        self.logger = logging.getLogger("infrastructure.wal")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(self.log_path, "a", buffering=1) # Line buffered

    def log_transaction(self, tx_type: str, data: dict):
        """
        Persist transaction to disk.
        Returns: tx_id
        """
        tx_id = time.time_ns()
        entry = {
            "id": tx_id,
            "ts": time.time(),
            "type": tx_type,
            "data": data
        }
        try:
            self._file.write(json.dumps(entry) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno()) # Force write to disk
            return tx_id
        except Exception as e:
            self.logger.critical(f"❌ WAL Write Failed! {e}")
            raise e

    def replay(self, handler_callback):
        """Replay log from start to restore state."""
        if not os.path.exists(self.log_path): return
        
        self.logger.info("↺ Replaying WAL...")
        count = 0
        with open(self.log_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    handler_callback(entry)
                    count += 1
                except Exception:
                    continue
        self.logger.info(f"✅ Replay Complete: {count} transactions.")

wal = WriteAheadLog()
