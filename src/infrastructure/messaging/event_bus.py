
import logging
import queue
import threading
import json
import time
from typing import Dict, Callable, Any

# v40 Infrastructure: Event Bus
# Implements a Publish-Subscribe pattern simulating a distributed message queue (ZeroMQ/Kafka style).
# Supports topic-based routing and async delivery.

class EventBus:
    def __init__(self):
        self._topics: Dict[str, list] = {}
        self._queue = queue.Queue()
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._logger = logging.getLogger("infrastructure.event_bus")
        self._worker_thread.start()

    def subscribe(self, topic: str, handler: Callable[[Dict], None]):
        """Subscribe a handler function to a specific topic."""
        if topic not in self._topics:
            self._topics[topic] = []
        self._topics[topic].append(handler)
        self._logger.debug(f"🔌 Subscribed to topic: {topic}")

    def publish(self, topic: str, payload: Dict):
        """Publish a message to a topic asynchronously."""
        message = {
            "topic": topic,
            "payload": payload,
            "timestamp": time.time_ns(),
            "id": f"msg_{time.time_ns()}"
        }
        self._queue.put(message)

    def _process_queue(self):
        while self._running:
            try:
                msg = self._queue.get(timeout=1.0)
                topic = msg["topic"]
                if topic in self._topics:
                    for handler in self._topics[topic]:
                        try:
                            # In a real distributed system, this would be an RPC or socket send
                            handler(msg["payload"])
                        except Exception as e:
                            self._logger.error(f"❌ Event Handler Error on {topic}: {e}")
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"❌ EventBus Critical Error: {e}")

    def stop(self):
        self._running = False
        self._worker_thread.join()

# Global Singleton for in-process simulation
event_bus = EventBus()
