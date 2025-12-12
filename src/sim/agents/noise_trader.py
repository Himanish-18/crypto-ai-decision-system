
import random
from src.sim.agents.base_agent import BaseAgent

# v41 Sim: Noise Trader
# Randomly takes liquidity (buys or sells) to create volume.

class NoiseTrader(BaseAgent):
    def __init__(self, activity_freq=0.5, size=0.01):
        super().__init__()
        self.activity_freq = activity_freq
        self.size = size

    def on_wakeup(self):
        if random.random() < 0.5:
            # Buy
            self.place_order("B", 1e9, self.size) # Market Buy (High Price)
        else:
            # Sell
            self.place_order("S", 0.0, self.size) # Market Sell (Low Price)
            
        # Re-schedule
        next_wakup = random.expovariate(1.0 / self.activity_freq)
        self.schedule_wakeup(next_wakup)

    def on_trade(self, trade_event):
        pass

    def on_ticker(self, ticker_event):
        pass
