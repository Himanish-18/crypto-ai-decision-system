
from src.sim.agents.base_agent import BaseAgent

# v41 Sim: Informed Agent
# Trades aggressively in the direction of a hidden "alpha" signal (simulated trend).

class InformedAgent(BaseAgent):
    def __init__(self, signal_function, aggression=1.0):
        super().__init__()
        self.signal_func = signal_function # Returns -1 to 1
        self.aggression = aggression

    def on_wakeup(self):
        signal = self.signal_func(self.bus.current_time)
        
        if abs(signal) > 0.5:
            side = "B" if signal > 0 else "S"
            # Larger size for stronger signal
            size = abs(signal) * self.aggression
            price = 1e9 if side == "B" else 0.0
            
            self.place_order(side, price, size)

        self.schedule_wakeup(1.0) # Check signal every second

    def on_trade(self, trade_event):
        pass

    def on_ticker(self, ticker_event):
        pass
