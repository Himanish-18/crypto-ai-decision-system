import heapq
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

logger = logging.getLogger("event_engine")

@dataclass(order=True)
class Event:
    timestamp: float
    priority: int # 0=High (MarketData), 1=Signal, 2=Order, 3=Fill
    payload: Union['MarketData', 'Signal', 'Order', 'Fill'] = field(compare=False)

@dataclass
class MarketData:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float

@dataclass
class Order:
    symbol: str
    side: str # 'BUY' / 'SELL'
    quantity: float
    order_type: str # 'LIMIT' / 'MARKET'
    price: float = 0.0
    id: str = ""

@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    price: float
    quantity: float
    cost: float
    timestamp: float

class EventBacktester:
    """
    v29 Institutional Event-Driven Engine.
    Simulates time-ordered event processing.
    """
    def __init__(self, start_time=0.0):
        self.current_time = start_time
        self.queue = []
        self.strategy = None
        self.exchange = None
        self.portfolio = None
    
    def push_event(self, event: Event):
        heapq.heappush(self.queue, event)
        
    def run(self):
        logger.info("🚀 Starting Event-Driven Backtest...")
        
        while self.queue:
            event = heapq.heappop(self.queue)
            self.current_time = event.timestamp
            
            # Dispatch
            if isinstance(event.payload, MarketData):
                self.on_market_data(event.payload)
            elif isinstance(event.payload, Order):
                # Order sent to exchange
                self.exchange.receive_order(event.payload, self.current_time)
            elif isinstance(event.payload, Fill):
                self.portfolio.on_fill(event.payload)
                
    def on_market_data(self, data: MarketData):
        # Update Exchange internal state (LOB)
        self.exchange.update_market(data, self.current_time)
        
        # Strategy decides
        if self.strategy:
            orders = self.strategy.on_market_data(data)
            for order in orders:
                # Add latency?
                latency = 0.020 # 20ms
                self.push_event(Event(self.current_time + latency, 2, order))

