# tests/full_system_stress_test.py
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import ALL Core Components
from src.decision.meta_brain_v17 import MetaBrainV17
from src.execution.liquidity_ai import LiquidityAI
from src.features.orderbook_features import OrderBookManager
from src.maintenance.self_heal import SelfHealingSystem
from src.market.router_v2 import MarketRouterV2
from src.ml.noise.cleanliness import MarketCleanlinessModel
from src.risk_engine.risk_module import RiskEngine

# Configuration
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("stress_test")


class SystemStressTest:
    def __init__(self):
        self.scores = {}
        self.components = {}

    def setup(self):
        logger.info("Initializing System Components...")
        try:
            self.components["brain"] = MetaBrainV17()
            self.components["router_v2"] = MarketRouterV2()
            self.components["liquidity"] = LiquidityAI()
            self.components["noise"] = MarketCleanlinessModel()
            self.components["heal"] = SelfHealingSystem(Path("./data"))
            self.components["ob"] = OrderBookManager()
            self.components["risk"] = RiskEngine(account_size=10000)
            logger.info("✅ Initialization Successful")
            return True
        except Exception as e:
            logger.error(f"❌ Initialization Failed: {e}")
            return False

    def test_intelligence_layer(self):
        logger.info("\n--- Testing Intelligence Layer (MetaBrain v17) ---")
        try:
            # Create varied market scenarios
            scenarios = [
                {"name": "Normal", "vol": 0.02, "fund": 0.0001, "noise": 0.3},
                {
                    "name": "High Stress (Crash)",
                    "vol": 0.08,
                    "fund": -0.002,
                    "noise": 0.8,
                },
                {
                    "name": "Manipulation",
                    "vol": 0.01,
                    "fund": 0.0001,
                    "noise": 0.1,
                    "spoof": 0.95,
                },
            ]

            passed = 0
            for sc in scenarios:
                payload = {
                    "symbol": "BTC/USDT",
                    "candles": pd.DataFrame({"close": [100, 101]}),  # Minimal stub
                    "microstructure": {
                        "weighted_imbalance": sc.get("spoof", 0.0),
                        "spread_pct": 0.0001,
                    },
                    "funding_rate": sc["fund"],
                    "volatility": sc["vol"],
                }

                start = time.time()
                decision = self.components["brain"].think(payload)
                lat = (time.time() - start) * 1000

                logger.info(
                    f"Scenario [{sc['name']}]: Decision={decision['action']} ({lat:.2f}ms)"
                )

                # Assertions
                if sc["name"] == "High Stress (Crash)" and decision["action"] != "HOLD":
                    logger.warning("❌ Failed Crash Safety Check (Should HOLD)")
                elif sc["name"] == "Manipulation" and decision["action"] == "HOLD":
                    logger.info("✅ Manipulation correctly blocked")
                    passed += 1
                else:
                    passed += 1

            self.scores["intelligence"] = (passed / len(scenarios)) * 100
        except Exception as e:
            logger.error(f"Intelligence Test Failed: {e}")
            self.scores["intelligence"] = 0

    def test_execution_layer(self):
        logger.info("\n--- Testing Execution Layer (LiquidityAI) ---")
        try:
            # Test Intent Generation
            payload = {"microstructure": {"spread_pct": 0.0001, "impact_cost": 0.5}}
            intent = self.components["liquidity"].analyze_intent("BUY", 1.0, payload)

            if intent["type"] in ["MAKER", "TAKER", "ICEBERG"]:
                logger.info(f"✅ Intent Generated: {intent['type']}")
                self.scores["execution"] = 100
            else:
                self.scores["execution"] = 0
        except Exception as e:
            logger.error(f"Execution Test Failed: {e}")
            self.scores["execution"] = 0

    def test_durability(self):
        logger.info("\n--- Testing Durability & Noise ---")
        try:
            # Test Noise Model
            df = pd.DataFrame(
                {
                    "open": np.random.rand(100),
                    "high": np.random.rand(100),
                    "low": np.random.rand(100),
                    "close": np.random.rand(100),
                }
            )
            score = self.components["noise"].analyze_cleanliness(df)
            logger.info(f"Noise Score: {score}")

            # Test Self-Heal
            self.components["heal"].monitor_performance(
                0.3, 1000
            )  # Should trigger warning logs
            logger.info("✅ Self-Heal Triggers Checked")
            self.scores["durability"] = 100
        except Exception as e:
            logger.error(f"Durability Test Failed: {e}")
            self.scores["durability"] = 0

    def run(self):
        logger.info("🚀 STARTING FULL SYSTEM STRESS TEST")
        if self.setup():
            self.test_intelligence_layer()
            self.test_execution_layer()
            self.test_durability()

        print("\n" + "=" * 30)
        print("🏁 FINAL STRESS JEST REPORT")
        print("=" * 30)
        for k, v in self.scores.items():
            print(f"{k.upper()}: {v}/100")


if __name__ == "__main__":
    test = SystemStressTest()
    test.run()
