
import random
from src.sim.agents.base_agent import BaseAgent

# v41 Sim: Liquidity Provider (Market Maker)
# Posts simulated liquidity around the mid-price.

class LiquidityProvider(BaseAgent):
    def __init__(self, spread_bps=10.0, size_range=(0.1, 1.0), update_freq=1.0):
        super().__init__()
        self.spread_bps = spread_bps
        self.size_range = size_range
        self.update_freq = update_freq
        self.active_orders = []
        self.mid_price = 10000.0 # Initial guess

    def on_wakeup(self):
        # Cancel old orders
        for oid in self.active_orders:
            self.cancel_order(oid)
        self.active_orders = []

        # Calculate levels
        half_spread = self.mid_price * (self.spread_bps / 10000.0) / 2
        bid_price = self.mid_price - half_spread
        ask_price = self.mid_price + half_spread
        
        # Place new orders (simple 1-level)
        size = random.uniform(*self.size_range)
        self.place_order("B", bid_price, size)
        
        size = random.uniform(*self.size_range)
        self.place_order("S", ask_price, size)
        
        self.schedule_wakeup(self.update_freq)

    def on_trade(self, trade_event):
        self.mid_price = trade_event['price']

    def on_ticker(self, ticker_event):
        if ticker_event['best_bid'] > 0 and ticker_event['best_ask'] < float('inf'):
            self.mid_price = (ticker_event['best_bid'] + ticker_event['best_ask']) / 2
