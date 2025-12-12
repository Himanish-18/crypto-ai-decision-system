import asyncio
import logging
from uuid import uuid4

from src.engine.event_bus import EventBus
from src.engine.events import OrderEvent, RiskCheckEvent

logger = logging.getLogger("service.execution")


class ExecutionGateway:
    """
    Service: Consumes Risk Decisions, routes Orders.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.bus.subscribe(RiskCheckEvent, self.handle_risk_decision)

    async def handle_risk_decision(self, event: RiskCheckEvent):
        """Execute if approved."""
        if not event.approved:
            # logger.warning(f"Signature Rejected: {event.reason}")
            return

        # logger.info(f"Executing Approved Signal: {event.signal_id}")

        # Emit Order Event (stub order)
        # Real logic would call Exchange API
        order = OrderEvent(
            event_id=str(uuid4()),
            symbol="BTC/USDT",
            order_type="MARKET",
            side="BUY",
            quantity=0.1,
        )
        # In a real system, we'd wait for fill confirmation
        # Here we just log
        logger.info(f"🚀 ORDER SENT: {order.side} {order.symbol}")
