import asyncio
import logging
from src.engine.event_bus import EventBus
from src.engine.events import FillEvent, OrderEvent

logger = logging.getLogger("service.portfolio")

class PortfolioEngine:
    """
    Service: Consumes Fills, tracks PnL and Positions.
    """
    def __init__(self, bus: EventBus):
        self.bus = bus
        # self.bus.subscribe(FillEvent, self.handle_fill) 
        # For now, subscribing to OrderEvent for demo flow, 
        # really should simulate Fill from Exchange
        self.bus.subscribe(OrderEvent, self.handle_order_ack)
        
    async def handle_order_ack(self, event: OrderEvent):
        """Update portfolio state on order (optimistic)"""
        # logger.info(f"Portfolio Update: Pending {event.side} {event.quantity}")
        pass
