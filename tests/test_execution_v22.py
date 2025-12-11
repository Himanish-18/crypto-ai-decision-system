import unittest
from src.execution.hft.fill_probability import FillProbabilityModel
from src.execution.hft.simulator import ExecutionSimulator
from src.execution.hft.router import SmartOrderRouter
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

class TestExecutionV22(unittest.TestCase):
    def setUp(self):
        self.fill_model = FillProbabilityModel()
        self.sim = ExecutionSimulator()
        self.router = SmartOrderRouter()
        
    def test_fill_prob(self):
        # Buying logic:
        # High OBI (Buy pressure) -> Price goes up -> Away from limit -> Lower Fill Prob
        prob_adverse = self.fill_model.predict("BUY", distance_pct=0.0, obi=0.8) # Strong buy pressure
        # prob = 1 / (1 + exp -(0.5 + 0 + 1.0*(-0.8))) = sigmoid(0.5 - 0.8) = sigmoid(-0.3) < 0.5
        
        prob_favorable = self.fill_model.predict("BUY", distance_pct=0.0, obi=-0.8) # Strong sell pressure (price drops to us)
        # prob = 1 / (1 + exp -(0.5 + 0 + 1.0*(0.8))) = sigmoid(1.3) > 0.5
        
        print(f"Prob Adverse: {prob_adverse:.2f}")
        print(f"Prob Favorable: {prob_favorable:.2f}")
        
        self.assertLess(prob_adverse, prob_favorable)
        
    def test_simulator(self):
        # Create Snapshots: Price zig zags 100 -> 101 -> 99
        snaps = [
            {"bids": [[100.0, 1.0]], "asks": [[100.5, 1.0]], "timestamp": 1},
            {"bids": [[100.2, 1.0]], "asks": [[100.7, 1.0]], "timestamp": 2}, # Price Up
            {"bids": [[99.0, 1.0]], "asks": [[99.5, 1.0]], "timestamp": 3}    # Price Down
        ]
        
        # Limit Buy at 100.0
        # Snap 1: Best Ask 100.5. No Fill.
        # Snap 2: Best Ask 100.7. No Fill.
        # Snap 3: Best Ask 99.5. 99.5 <= 100.0. Fill!
        
        orders = [{"side": "BUY", "price": 100.0, "qty": 0.1, "type": "LIMIT"}]
        res = self.sim.run_simulation(snaps, orders)
        
        print(f"Sim Result: {res}")
        self.assertEqual(res["fill_rate"], 1.0)
        self.assertEqual(res["fills"][0]["fill_price"], 100.0) # Limit filled at limit price? 
        # Actually logic said: fill_price = target_price. Yes.
        
    def test_router_adaptive(self):
        # Case 1: High Vol, Adverse OBI -> Expect Taker
        market_data = {"ob_imbalance": 0.9, "volatility": 0.01} # Buy OBI=0.9 (bad for buy limit)
        route = self.router.route_order("BUY", 10.0, market_data)
        print(f"Route 1 (High Vol/Bad OBI): {route['type']}")
        self.assertEqual(route["type"], "MARKET")
        if route["qty"] < 10.0:
            print("  Sliced correctly.")
        
        # Case 2: Low Vol, Good OBI -> Expect Maker
        market_data_calm = {"ob_imbalance": -0.5, "volatility": 0.001} # Sell OBI (Price drops)
        route2 = self.router.route_order("BUY", 0.5, market_data_calm)
        print(f"Route 2 (Calm/Good OBI): {route2['type']}")
        self.assertEqual(route2["type"], "LIMIT")

if __name__ == "__main__":
    unittest.main()
