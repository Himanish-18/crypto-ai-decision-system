
import subprocess
import time
import os
import signal
import sys
import logging
from datetime import datetime

# Setup Watchdog Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [WATCHDOG] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("watchdog.log"), logging.StreamHandler()]
)
logger = logging.getLogger("watchdog")

WATCH_FILE = "live_trading.log"
TIMEOUT_SEC = 30
CMD = [".venv/bin/python3", "-m", "src.main", "--mode=institutional", "--execution=native", "--backtest=vector"]

class SystemWatchdog:
    def __init__(self):
        self.process = None
        self.last_check_time = time.time()
        
    def start_process(self):
        logger.info(f"🚀 Launching System: {' '.join(CMD)}")
        self.process = subprocess.Popen(CMD)
        return self.process

    def kill_process(self):
        if self.process:
            logger.warning(f"💀 Killing PID {self.process.pid}...")
            try:
                os.kill(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process = None

    def get_file_mtime(self, filepath):
        if not os.path.exists(filepath):
            return 0
        return os.path.getmtime(filepath)

    def monitor(self):
        logger.info("👀 Watchdog Started. Monitoring Heartbeat...")
        
        # Start initial process
        self.start_process()
        
        while True:
            time.sleep(5)
            
            # 1. Check if process is alive
            if self.process.poll() is not None:
                logger.error("⚠️ Process died unexpectedly! Restarting...")
                self.start_process()
                continue
            
            # 2. Check Log Heartbeat
            last_mod = self.get_file_mtime(WATCH_FILE)
            now = time.time()
            gap = now - last_mod
            
            if gap > TIMEOUT_SEC:
                logger.error(f"❄️ FROZEN DETECTED! No log update for {gap:.1f}s (Limit: {TIMEOUT_SEC}s).")
                self.restart_system()
            else:
                # User Feedback: Log heartbeat periodically to show ALIVENESS
                # Log every 30 seconds to avoid excessive logging
                if int(now) % 30 == 0: 
                    logger.info(f"💓 Heartbeat OK. Gap: {gap:.1f}s | System Healthy.")

if __name__ == "__main__":
    try:
        dog = SystemWatchdog()
        dog.monitor()
    except KeyboardInterrupt:
        logger.info("🛑 Watchdog Stopped by User.")
        if dog.process:
            dog.kill_process()
