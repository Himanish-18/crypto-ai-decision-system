import time
import logging
import asyncio
import numpy as np
from typing import List, Dict

# Setup Logging
logger = logging.getLogger("latency_manager")

class LatencyManager:
    """
    Monitors API latency and determines execution style.
    
    Rules:
    - Latency < 150ms: Prefer MAKER (Passive) to capture spread.
    - Latency >= 150ms: Prefer TAKER (Aggressive) to ensure fill.
    """
    def __init__(self, window_size: int = 50):
        self.latencies: List[float] = []
        self.window_size = window_size
        self.maker_threshold_ms = 150.0
        
    def record_latency(self, start_time: float):
        """Record the round-trip time of an API call."""
        latency_ms = (time.time() - start_time) * 1000.0
        self.latencies.append(latency_ms)
        
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
            
    def get_avg_latency(self) -> float:
        """Get average latency over the rolling window."""
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies)
        
    def get_execution_style(self) -> str:
        """
        Determine execution style based on current latency conditions.
        Returns: "PASSIVE" or "AGGRESSIVE"
        """
        avg_latency = self.get_avg_latency()
        
        if avg_latency < self.maker_threshold_ms:
            return "PASSIVE"
        else:
            logger.warning(f"🐢 High Latency detected ({avg_latency:.2f}ms). Switching to AGGRESSIVE.")
            return "AGGRESSIVE"

    async def measure_ping(self, exchange_client):
        """
        Proactively measure latency by pinging the exchange.
        """
        try:
            start = time.time()
            # Assuming exchange_client has a fetch_time or similar lightweight method
            await exchange_client.fetch_time()
            self.record_latency(start)
        except Exception as e:
            logger.error(f"Ping failed: {e}")

# Mock for Verification
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    lm = LatencyManager()
    
    # Simulate Low Latency
    for _ in range(10):
        start = time.time()
        time.sleep(0.05) # 50ms
        lm.record_latency(start)
        
    logger.info(f"Avg Latency: {lm.get_avg_latency():.2f}ms. Style: {lm.get_execution_style()}")
    
    # Simulate High Latency spike
    for _ in range(10):
        start = time.time()
        time.sleep(0.2) # 200ms
        lm.record_latency(start)
        
    logger.info(f"Avg Latency: {lm.get_avg_latency():.2f}ms. Style: {lm.get_execution_style()}")
