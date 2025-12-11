import unittest
import pandas as pd
import numpy as np
from src.risk.portfolio_risk_v3 import PortfolioRiskEngine
from src.risk.scenarios import ScenarioSimulator
from src.risk.hedger import DynamicHedger

class TestPortfolioRiskV3(unittest.TestCase):
    def setUp(self):
        self.engine = PortfolioRiskEngine()
        self.simulator = ScenarioSimulator()
        self.hedger = DynamicHedger()
        
        # Mock Data
        dates = pd.date_range("2024-01-01", periods=100)
        self.history = pd.DataFrame({
            "BTC": np.random.normal(0, 0.02, 100),
            "ETH": np.random.normal(0, 0.03, 100)
        }, index=dates)
        
    def test_var_calc(self):
        positions = {"BTC": 1.0, "ETH": 10.0} # 1 BTC, 10 ETH
        prices = {"BTC": 50000, "ETH": 3000}
        equity = 100000
        
        self.engine.update_portfolio(positions, prices, equity)
        metrics = self.engine.calculate_risk_metrics(self.history)
        
        print(f"VaR 99: {metrics['var_99']:.4f}")
        self.assertGreater(metrics["var_99"], 0.0)
        self.assertGreater(metrics["es_99"], metrics["var_99"])
        
    def test_scenarios(self):
        positions = {"BTC": 1.0}
        prices = {"BTC": 50000}
        self.engine.update_portfolio(positions, prices, 50000)
        
        res = self.simulator.run_all(self.engine.positions)
        print("Scenario Results:", res)
        # BTC Crash 10% on 50k pos = -5000
        self.assertAlmostEqual(res["crypto_crash_10"], -5000.0)
        
    def test_hedger(self):
        metrics = {"net_delta": 10000.0} # Long 10k
        trades = self.hedger.propose_hedge(metrics, {})
        
        self.assertTrue(len(trades) > 0)
        self.assertEqual(trades[0]["side"], "SELL")
        self.assertAlmostEqual(trades[0]["amount_usd"], 10000.0)

if __name__ == "__main__":
    unittest.main()
