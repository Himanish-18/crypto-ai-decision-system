
import random
import logging

# v39 Probabilistic SOR
# Simulates routing across Binance, Bybit, OKX based on fill probability.

class ProbabilisticSOR:
    def __init__(self):
        self.venues = ["Binance", "Bybit", "OKX"]
        self.fill_probs = {"Binance": 0.95, "Bybit": 0.85, "OKX": 0.80}
        self.logger = logging.getLogger("sor")
        
    def route_order(self, size, price):
        # Rank venues by probabilty adjusted cost (simplification: just prob)
        ranked = sorted(self.venues, key=lambda v: self.fill_probs[v], reverse=True)
        
        best_venue = ranked[0]
        self.logger.info(f"SOR Routing {size} @ {price} to {best_venue} (Prob: {self.fill_probs[best_venue]})")
        
        return best_venue

    def simulate_outcome(self, venue):
        if random.random() < self.fill_probs[venue]:
            return "FILLED"
        else:
            return "REJECTED"
