
import heapq
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Dict

# v41 Sim: Deterministic Event Bus
# Uses a Priority Queue to process events strictly in timestamp order.

@dataclass(order=True)
class SimEvent:
    timestamp: float
    priority: int
    topic: str = field(compare=False)
    payload: Any = field(compare=False)

class SimEventBus:
    def __init__(self):
        self._queue = []
        self._subscribers: Dict[str, List[Callable]] = {}
        self.current_time = 0.0
        self.event_count = 0

    def subscribe(self, topic: str, handler: Callable):
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)

    def publish(self, topic: str, payload: Any, delay: float = 0.0, priority: int = 10):
        """
        Schedule an event to happen at current_time + delay.
        Lower priority value = processed earlier if timestamps match.
        """
        exec_time = self.current_time + delay
        event = SimEvent(exec_time, priority, topic, payload)
        heapq.heappush(self._queue, event)

    def run_until(self, end_time: float) -> int:
        """Process events until sim time reaches end_time."""
        processed = 0
        last_log_time = -1
        while self._queue and self._queue[0].timestamp <= end_time:
            event = heapq.heappop(self._queue)
            self.current_time = event.timestamp
            
            if event.topic in self._subscribers:
                for handler in self._subscribers[event.topic]:
                    handler(event.payload)
            
            processed += 1
            self.event_count += 1
            
            # Progress Log
            if int(self.current_time) > int(last_log_time):
                print(f"  [SimTime: {int(self.current_time)}s] Events: {self.event_count}")
                last_log_time = int(self.current_time)
        
        self.current_time = end_time
        return processed

    def reset(self):
        self._queue = []
        self.current_time = 0.0
        self.event_count = 0

# Global Sim Context (for easy access within simulation modules)
sim_bus = SimEventBus()
