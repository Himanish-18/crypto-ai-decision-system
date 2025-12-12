import asyncio
import logging
from uuid import uuid4

from src.engine.event_bus import EventBus
from src.engine.events import RiskCheckEvent, SignalEvent

logger = logging.getLogger("service.risk")


class RiskVetoService:
    """
    Service: Consumes Signals, checks Limits, emits RiskDecision.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.bus.subscribe(SignalEvent, self.handle_signal)

    async def handle_signal(self, event: SignalEvent):
        """Vet the signal."""
        # logger.info(f"Risk Check: {event.source} wanted {event.direction}")

        # Stub: Risk Check logic
        # Real logic would invoke VarEngine / HardVeto

        approved = True
        reason = "Pass"

        risk_event = RiskCheckEvent(
            event_id=str(uuid4()),
            signal_id=event.event_id,
            approved=approved,
            adjusted_size=1.0,  # Full size
            reason=reason,
        )
        await self.bus.publish(risk_event)
