import logging
import time
import signal
import sys
from typing import List

import logging
import time
import signal
import sys
from typing import List

from src.guardian.monitors.base_monitor import BaseMonitor, Anomaly
from src.guardian.monitors.log_monitor import LogMonitor
from src.guardian.monitors.logic_monitor import LogicMonitor
from src.guardian.monitors.resource_monitor import ResourceMonitor
from src.guardian.healer import SentinelHealer

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SENTINEL] - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sentinel.log"), logging.StreamHandler()]
)
logger = logging.getLogger("sentinel.core")

class Sentinel:
    """
    v43 Autonomous Sentinel Daemon.
    The 'Self-Driving' Reliability Engineer.
    """
    def __init__(self):
        self.monitors: List[BaseMonitor] = []
        self.healer = SentinelHealer()
        self.running = False
        
        # --- Register Default Monitors ---
        self.register_monitor(LogMonitor("live_trading.log"))
        self.register_monitor(LogMonitor("watchdog.log"))
        self.register_monitor(LogicMonitor("predictions.log"))
        self.register_monitor(ResourceMonitor())

    def register_monitor(self, monitor: BaseMonitor):
        self.monitors.append(monitor)
        logger.info(f"👁️ Monitor Registered: {monitor.name}")

    def run_once(self):
        """
        Single pass of all monitors.
        """
        for monitor in self.monitors:
            try:
                anomalies = monitor.check()
                if anomalies:
                    for anomaly in anomalies:
                        logger.warning(f"🚨 ANOMALY DETECTED: {anomaly.type}")
                        self.healer.heal(anomaly)
            except Exception as e:
                logger.error(f"❌ Monitor {monitor.name} failed: {e}", exc_info=True)

    def run_forever(self, interval=5.0):
        """
        Main Sentinel Loop.
        """
        self.running = True
        logger.info("🛡️ SENTINEL SYSTEM ACTIVATED. Guarding Project...")
        
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        while self.running:
            self.run_once()
            time.sleep(interval)

    def shutdown(self, signum, frame):
        logger.info("🛑 Sentinel Stopping...")
        self.running = False
        sys.exit(0)

if __name__ == "__main__":
    # Test Run
    sentinel = Sentinel()
    # Stub Monitor for Test
    # sentinel.register_monitor(StubMonitor())
    sentinel.run_forever()
