import unittest
import torch
import pandas as pd
import numpy as np
from src.ml.uncertainty import UncertaintyEngine
from src.ml.meta_label_v2 import TripleBarrierLabeler, LossProbabilityModelV2
from src.ml.adversarial import AdversarialRobustness
from src.ml.moe_router import MoERouter

class TestMLv18(unittest.TestCase):
    def test_uncertainty_engine(self):
        print("\nTesting Uncertainty Engine (MC Dropout)...")
        # Mock model
        model = torch.nn.Linear(10, 1)
        eng = UncertaintyEngine(model, n_passes=5)
        x = torch.randn(1, 10)
        mean, epi, ale = eng.predict_with_uncertainty(x)
        print(f"Mean: {mean:.4f}, Epistemic: {epi:.4f}")
        self.assertIsInstance(mean, float)
        self.assertGreaterEqual(epi, 0.0)

    def test_meta_labeling(self):
        print("\nTesting Triple Barrier Labeling...")
        dates = pd.date_range("2023-01-01", periods=100, freq="1min")
        prices = pd.Series(np.random.normal(100, 1, 100).cumsum(), index=dates)
        
        lbl = TripleBarrierLabeler(time_limit=10)
        labels = lbl.get_labels(prices)
        print(f"Labels Generated: {labels.value_counts().to_dict()}")
        self.assertEqual(len(labels), 100)
        
    def test_adversarial_check(self):
        print("\nTesting Adversarial Robustness...")
        adv = AdversarialRobustness()
        # Stub check since we don't pass a real trained model
        score = adv.check_stability(None, torch.randn(1, 10))
        print(f"Stability Score (Stub): {score}")
        self.assertEqual(score, 1.0)

    def test_moe_router(self):
        print("\nTesting MoE Router...")
        router = MoERouter()
        res = router.route_predict("RISK_ON", None)
        print(f"Expert: {res['expert']}, Signal: {res['signal']}")
        self.assertEqual(res['expert'], "TrendFollower")
        
        res2 = router.route_predict("LIQ_CRUNCH", None)
        self.assertEqual(res2['expert'], "ShortSeller")

if __name__ == '__main__':
    unittest.main()
