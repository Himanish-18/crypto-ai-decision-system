
import random
import time

# v39 Network Jitter Simulator
# Injects random latency to simulate real-world internet conditions.

class JitterSim:
    def __init__(self, base_latency_ms=20, jitter_std=50):
        self.base_latency = base_latency_ms
        self.jitter_std = jitter_std
        
    def delay(self):
        """
        Sleeps for a random duration drawn from a LogNormal distribution.
        """
        # Latency cannot be negative
        lat = self.base_latency + max(0, random.gauss(0, self.jitter_std))
        
        # In Microseconds for simulation
        # time.sleep(lat / 1000.0) 
        return lat

    def get_latency(self):
        return self.delay()
