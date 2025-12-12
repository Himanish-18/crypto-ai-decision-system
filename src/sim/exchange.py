
from src.sim.event_bus import sim_bus
from src.sim.orderbook_l3 import SimOrderBookL3, SimOrder
import uuid

# v41 Sim: Exchange Coordinator
# Connects Agents to the OrderBook and EventBus.

class SimExchange:
    def __init__(self):
        self.book = SimOrderBookL3()
        self.bus = sim_bus
        self.bus.subscribe("order.new", self._on_new_order)
        self.bus.subscribe("order.cancel", self._on_cancel_order)

    def _on_new_order(self, payload):
        """payload: {owner_id, side, price, size, type}"""
        order_id = str(uuid.uuid4())[:8]
        order = SimOrder(
            order_id=order_id,
            side=payload['side'],
            price=payload['price'],
            size=payload['size'],
            entry_time=sim_bus.current_time,
            owner_id=payload['owner_id']
        )
        
        fills = self.book.add_order(order)
        
        # Publish Fills
        for fill in fills:
            self.bus.publish("market.trade", fill)
            
        # Publish Quote Update
        self._publish_ticker()

    def _on_cancel_order(self, payload):
        """payload: {order_id}"""
        success = self.book.cancel_order(payload['order_id'])
        if success:
             self._publish_ticker()

    def _publish_ticker(self):
        self.bus.publish("market.ticker", self.book.get_snapshot())

exchange_sim = SimExchange()
