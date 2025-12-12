
from abc import ABC, abstractmethod
from src.sim.event_bus import sim_bus
import uuid

# v41 Sim: Base Agent Interface
# All simulation agents must inherit from this.

class BaseAgent(ABC):
    def __init__(self, agent_id=None):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.bus = sim_bus
        self.bus.subscribe("market.trade", self.on_trade)
        self.bus.subscribe("market.ticker", self.on_ticker)
        self.start_delay = 0

    def start(self, delay: float = 0.0):
        self.start_delay = delay
        self.bus.publish("agent.start", {"agent_id": self.agent_id}, delay=delay)
        # Schedule first wakeup
        self.schedule_wakeup(delay)

    def schedule_wakeup(self, delay: float):
        self.bus.publish("agent.wakeup", {"agent_id": self.agent_id}, delay=delay)
        self.bus.subscribe("agent.wakeup", self._on_wakeup_internal)

    def _on_wakeup_internal(self, payload):
        if payload['agent_id'] == self.agent_id:
            self.on_wakeup()

    @abstractmethod
    def on_wakeup(self):
        """Called periodically or when scheduled."""
        pass

    @abstractmethod
    def on_trade(self, trade_event):
        """Called on every public trade."""
        pass

    @abstractmethod
    def on_ticker(self, ticker_event):
        """Called on orderbook update."""
        pass

    def place_order(self, side: str, price: float, size: float):
        self.bus.publish("order.new", {
            "owner_id": self.agent_id,
            "side": side,
            "price": price,
            "size": size
        })

    def cancel_order(self, order_id: str):
        self.bus.publish("order.cancel", {
            "owner_id": self.agent_id,
            "order_id": order_id
        })
