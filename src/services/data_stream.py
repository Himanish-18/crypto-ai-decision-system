import asyncio
import logging
import random
from src.engine.event_bus import EventBus
from src.engine.events import MarketTick
from uuid import uuid4

logger = logging.getLogger("service.data")

class DataStreamService:
    """
    Service: Ingests market data and publishes updates.
    """
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.running = False
        
    async def start(self):
        self.running = True
        logger.info("Starting Data Stream Service...")
        asyncio.create_task(self._poll_market())
        
    async def _poll_market(self):
        """Simulate polling market data (replace with real WS in prod)"""
        while self.running:
            # Stub: Simulate BTC tick
            price = 90000 + (random.random() * 100)
            
            tick = MarketTick(
                event_id=str(uuid4()),
                symbol="BTC/USDT",
                price=price,
                volume=1.5,
                bid=price-5,
                ask=price+5,
                exchange="BINANCE"
            )
            
            # await self.bus.publish(tick) # High frequency noise
            # For verification, publish periodically
            await self.bus.publish(tick)
            
            await asyncio.sleep(1) # 1Hz for test, remove for prod
            
    def stop(self):
        self.running = False
