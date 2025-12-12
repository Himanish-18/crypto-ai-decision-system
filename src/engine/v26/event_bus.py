
import logging
from typing import Callable, List, Dict
from collections import defaultdict
import json

logger = logging.getLogger("kafka_bus_stub")

class KafkaEventBus:
    """
    In-memory Event Bus mimicking Kafka topics.
    """
    
    _INSTANCE = None
    
    def __init__(self):
        self.topics: Dict[str, List[Callable]] = defaultdict(list)
        
    @classmethod
    def get_instance(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = KafkaEventBus()
        return cls._INSTANCE

    def subscribe(self, topic: str, handler: Callable):
        self.topics[topic].append(handler)
        logger.info(f"Subscribed to {topic}")

    def publish(self, topic: str, message: Dict):
        """
        Synchronous publish for simplicity in V1 Stub.
        """
        payload = json.dumps(message)
        subscribers = self.topics.get(topic, [])
        for sub in subscribers:
            try:
                sub(json.loads(payload))
            except Exception as e:
                logger.error(f"Error processing message on {topic}: {e}")
