import logging
import sys
import unittest

import pandas as pd

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
import shutil
from pathlib import Path

import numpy as np

from src.decision.meta_brain_v21 import MetaBrainV21
from src.models.regime_ensemble import RegimeEnsemble


class TestRegimeV21(unittest.TestCase):
    def setUp(self):
        # Create Dummy Data
        self.data_dir = Path("data/regime_test")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.data_dir / "regime_ensemble.pkl"

        # Train Dummy Model
        # X: [vol, funding, iv]
        # Y: 1 (Crash) if vol > 2.0 (Adjusted for realistic calculation)
        X = pd.DataFrame(
            {
                "volatility": np.random.uniform(0, 10, 100),
                "funding_rate": np.random.uniform(-0.01, 0.01, 100),
                "iv_index": np.random.uniform(20, 100, 100),
            }
        )
        y = (X["volatility"] > 3.0).astype(int)  # Trigger if > 3%

        self.ensemble = RegimeEnsemble(n_components=2)
        # Mock xgb if not installed
        if self.ensemble.xgb_model is None:
            # Just leave it None, predict will return 0 unless HMM works
            pass

        # Train
        self.ensemble.fit(X, y)
        self.ensemble.save(self.model_path)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_hard_veto(self):
        print(f"DEBUG: MetaBrainV21 source: {MetaBrainV21}")
        # Init Brain
        brain = MetaBrainV21(str(self.model_path))

        # 1. Low Risk Scenario
        candle_safe = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp.now(),
                    "close": 50000.0,
                    "open": 50000.0,
                    "high": 50100.0,
                    "low": 49900.0,
                    "volume": 100,
                }
            ]
        )
        # We need more history for std deviation usually, but MetaBrainV21
        # calculates vol from pct_change of history.
        # Let's provide a history DF
        hist_safe = pd.DataFrame({"close": np.random.normal(50000, 10, 100)})  # Low vol

        # Mock macro feeds to return safe
        brain.macro_feeds.fetch_live_metrics = lambda: {
            "funding_rate": 0.0001,
            "iv_index": 30,
        }

        # Pass dict as expected by base classes
        decision = brain.think(
            {"candles": hist_safe, "regime": "NEUTRAL", "action": "HOLD"}
        )
        print(f"Safe Decision: {decision}")
        # Expect Normal (Not Vetoed by Regime, though might be HOLD by other logic)
        self.assertNotEqual(decision.get("veto_reason"), "REGIME_RISK")

        # 2. High Risk Scenario
        hist_risk = pd.DataFrame(
            {"close": np.random.normal(50000, 2000, 100)}  # High vol
        )
        # Mock metrics to high risk
        brain.macro_feeds.fetch_live_metrics = lambda: {
            "funding_rate": -0.005,
            "iv_index": 90,
        }

        # Force XGB prediction high if mocking fails
        if not brain.regime_ensemble.xgb_model:
            # Manually inject a risk score for testing logic if XGB missing
            brain.regime_ensemble.predict_risk = lambda x: (0.9, {"mock": True})

        decision_risk = brain.think({"candles": hist_risk, "regime": "NEUTRAL"})
        print(f"Risk Decision: {decision_risk}")

        self.assertEqual(decision_risk.get("veto_reason"), "REGIME_RISK")
        self.assertEqual(decision_risk.get("action"), "HOLD")


if __name__ == "__main__":
    unittest.main()
