import asyncio
import logging
from typing import Callable, Dict, List, Type
from collections import defaultdict
from src.engine.events import Event

logger = logging.getLogger("event_bus")

class EventBus:
    """
    Asyncio-based Event Bus for Microservices Architecture.
    Supports pub/sub pattern.
    """
    def __init__(self):
        self._subscribers: Dict[Type[Event], List[Callable]] = defaultdict(list)
        self._queue = asyncio.Queue()
        self._running = False
        
    def subscribe(self, event_type: Type[Event], handler: Callable):
        """Register a handler for a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed {handler.__name__} to {event_type.__name__}")
        
    async def publish(self, event: Event):
        """Publish an event to the queue."""
        await self._queue.put(event)
        
    async def run(self):
        """Main Event Loop Processor."""
        self._running = True
        logger.info("⚡ Event Bus Started")
        
        while self._running:
            try:
                event = await self._queue.get()
                event_type = type(event)
                
                if event_type in self._subscribers:
                    handlers = self._subscribers[event_type]
                    # Execute handlers concurrently
                    tasks = [handler(event) for handler in handlers]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                self._queue.task_done()
                
            except Exception as e:
                logger.error(f"Event Bus Error: {e}", exc_info=True)
                
    def stop(self):
        self._running = False
        logger.info("Event Bus Stopping...")
