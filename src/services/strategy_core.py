import asyncio
import logging
from src.engine.event_bus import EventBus
from src.engine.events import MarketTick, SignalEvent
from uuid import uuid4

logger = logging.getLogger("service.strategy")

class StrategyCoreService:
    """
    Service: Consumes MarketTick, runs Strategy, emits Signal.
    """
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.bus.subscribe(MarketTick, self.handle_tick)
        
    async def handle_tick(self, event: MarketTick):
        """Process incoming tick."""
        # logger.debug(f"Strategy Process: {event.symbol} @ {event.price}")
        
        # Stub: Simple logic
        # Real logic would invoke MetaBrain here
        
        # Emit Signal
        sig = SignalEvent(
            event_id=str(uuid4()),
            symbol=event.symbol,
            signal_type="ALPHA",
            direction="BUY",
            strength=0.8,
            confidence=0.9,
            source="MetaBrainStub"
        )
        
        if event.price > 0: # Always true stub
            await self.bus.publish(sig)
