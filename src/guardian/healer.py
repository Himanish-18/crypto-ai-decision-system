import logging
import time
from typing import Dict, Any

# Fix circular import if generic anomaly needed, but for now we pass data
from src.guardian.monitors.base_monitor import Anomaly 

logger = logging.getLogger("sentinel.healer")

class SentinelHealer:
    """
    The 'Doctor' of the system. Receives Anomalies and prescribes/executes Fixes.
    """
    def __init__(self):
        self.action_history = []

    def heal(self, anomaly: Anomaly):
        """
        Decide and execute a fix for the given anomaly.
        """
        logger.info(f"🚑 HEALER RECEIVED: [{anomaly.severity}] {anomaly.type}")

        if anomaly.type == "test_anomaly":
            logger.info("🧪 Test Anomaly detected. No action needed.")
            return

        if anomaly.type == "NEUTRAL_LOCK":
            logger.warning("🔓 Neutral Lock Detected. ACTIONS: Relax Threshold + Restart.")
            self.relax_noise_threshold()
            self.restart_service()
            return

        if anomaly.type == "MEMORY_LEAK":
             logger.warning(f"💧 Memory Leak ({anomaly.details.get('used_mb')}MB). ACTION: Restart.")
             self.restart_service()
             return

        logger.warning(f"⚠️ No specific heal action defined for {anomaly.type} yet.")

    def restart_service(self):
        """
        Restarts the main trading process.
        """
        import os
        import signal
        import subprocess
        
        # 1. Find and Kill
        # For simplicity in this v43, we assume we might need to kill the main.py or the watchdog.
        # Actually, if Sentinel runs separately, it should kill the *other* python processes.
        # We rely on 'pkill -f src.main' style for now or use psutil if available.
        logger.info("♻️ Executing Service Restart...")
        try:
            subprocess.run(["pkill", "-f", "src.main"])
            time.sleep(2)
            # 2. Restart via Watchdog (which monitors and restarts src.main)
            # If watchdog is running, it will auto-restart src.main when it dies.
            # If watchdog itself is dead, we might need to start it.
            # Let's check if watchdog is running.
            status = subprocess.run(["pgrep", "-f", "watchdog.py"], capture_output=True)
            if status.returncode != 0:
                logger.info("🚀 Watchdog not found. Starting Watchdog...")
                subprocess.Popen([".venv/bin/python3", "src/guardian/watchdog.py"])
            else:
                 logger.info("✅ Watchdog is active, it should revive src.main.")
        except Exception as e:
            logger.error(f"Restart Failed: {e}")

    def relax_noise_threshold(self):
        """
        Hot-patch the noise immunity threshold.
        """
        target_file = "src/ml/noise/cleanliness.py"
        logger.info(f"🛠️ Relaxing Noise Threshold in {target_file}...")
        try:
            with open(target_file, 'r') as f:
                content = f.read()
            
            # Simple replace of hardcoded value if it's low
            # Looking for "self.noise_threshold = 0.65" or similar
            # We blindly replace 0.65 -> 0.85, 0.70 -> 0.85 etc?
            # Or just set it to 0.90 to be safe.
            import re
            new_content = re.sub(r"self\.noise_threshold\s*=\s*0\.\d+", "self.noise_threshold = 0.90", content)
            
            if content != new_content:
                with open(target_file, 'w') as f:
                    f.write(new_content)
                logger.info("✅ Threshold relaxed to 0.90")
            else:
                logger.info("⚠️ Threshold already high or not found.")
        except Exception as e:
            logger.error(f"Patch Failed: {e}")

