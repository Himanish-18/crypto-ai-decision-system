
import logging
import time
from abc import ABC, abstractmethod
from src.infrastructure.messaging.event_bus import event_bus

# v40 Architecture: Base Microservice
# Standardizes lifecycle, logging, and event handling for all services.

class MicroService(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"service.{name}")
        self.running = False
        self.bus = event_bus

    def start(self):
        self.logger.info(f"🚀 Starting Service: {self.name}")
        self.running = True
        self.on_start()
        
    def stop(self):
        self.logger.info(f"🛑 Stopping Service: {self.name}")
        self.running = False
        self.on_stop()

    @abstractmethod
    def on_start(self):
        """Lifecycle hook for startup logic"""
        pass

    @abstractmethod
    def on_stop(self):
        """Lifecycle hook for shutdown logic"""
        pass

    def publish(self, topic: str, data: dict):
        self.bus.publish(topic, data)

    def subscribe(self, topic: str, handler):
        self.bus.subscribe(topic, handler)
