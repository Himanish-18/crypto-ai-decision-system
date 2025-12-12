
import random
from collections import deque

# v39 Synthetic L3 Simulator
# Simulates full depth messages for testing execution logic without pcap.

class L3Simulator:
    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.msg_queue = deque()
        
    def generate_noise(self, mid_price=50000.0):
        # Generate random adds/cancels around mid
        event_type = random.choice(["ADD", "CANCEL", "TRADE"])
        
        if event_type == "ADD":
            side = random.choice(["B", "S"])
            price = mid_price + random.uniform(-10, 10)
            size = random.uniform(0.1, 2.5)
            self.msg_queue.append({"type": "ADD", "side": side, "price": price, "size": size})
            
        elif event_type == "TRADE":
             # Simulate a sweep
             self.msg_queue.append({"type": "TRADE", "size": random.uniform(1.0, 5.0)})
             
    def get_next_event(self):
        if self.msg_queue:
            return self.msg_queue.popleft()
        return None
