from collections import deque
from .engine import Event, Fill, Order, MarketData
import logging

logger = logging.getLogger("exchange_sim")

class ExchangeSimulator:
    """
    Simulates a matching engine with FIFO queues and Latency.
    """
    def __init__(self, engine, latency_ms=10.0):
        self.engine = engine
        self.latency_s = latency_ms / 1000.0
        
        # Order Books: symbol -> side -> deque of Orders
        self.bids = {}
        self.asks = {}
        self.current_market = {} # symbol -> MarketData
        
    def update_market(self, data: MarketData, timestamp: float):
        self.current_market[data.symbol] = data
        # Check matching for existing limit orders
        self._match_orders(data.symbol, timestamp)

    def receive_order(self, order: Order, timestamp: float):
        # Enqueue with latency
        # In this simple model, we process immediately but fill timestamp is delayed?
        # Or we just add to book.
        
        if order.order_type == "MARKET":
            self._execute_market_order(order, timestamp)
        else:
            self._add_limit_order(order, timestamp)
            
    def _execute_market_order(self, order, timestamp):
        mkt = self.current_market.get(order.symbol)
        if not mkt: return # No price yet
        
        price = mkt.ask if order.side == "BUY" else mkt.bid
        # Slippage model could go here
        
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=order.quantity,
            cost=0.001 * price, # Fee
            timestamp=timestamp + self.latency_s
        )
        self.engine.push_event(Event(fill.timestamp, 3, fill))
        
    def _add_limit_order(self, order, timestamp):
        # Add to local book
        pass # Simplified for v29 demo
        
    def _match_orders(self, symbol, timestamp):
        # Match limit orders against new price
        pass
