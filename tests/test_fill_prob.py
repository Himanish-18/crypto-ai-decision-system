
import unittest
import pandas as pd
import numpy as np
import os
from src.execution.fill_probability_model import FillProbabilityModel

class TestFillProbabilityModel(unittest.TestCase):
    def setUp(self):
        self.model = FillProbabilityModel()
        
    def test_init(self):
        self.assertIsNotNone(self.model.model)
        
    def test_training_and_prediction(self):
        # Create dummy data
        X = pd.DataFrame({
            "spread_bps": np.random.rand(100),
            "depth_skew": np.random.rand(100) - 0.5,
            "volatility_1m": np.random.rand(100) * 0.05,
            "trade_flow_imbalance": np.random.rand(100) - 0.5,
            "dist_to_mid": np.random.rand(100) * 10
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        self.model.train(X, y)
        
        # Test predict
        state = {
            "spread_bps": 5.0,
            "depth_skew": 0.2,
            "volatility_1m": 0.01,
            "trade_flow_imbalance": 0.1,
            "dist_to_mid": 0.0
        }
        prob = self.model.predict_fill_prob(state)
        self.assertTrue(0.0 <= prob <= 1.0)
        print(f"Test Predicted Prob: {prob}")

    def test_save_load(self):
        # Train with mixed classes
        X = pd.DataFrame(np.random.rand(10, 5), columns=self.model.feature_cols)
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.model.train(X, y)
        
        # Save
        self.model.save("test_model.pkl")
        
        # Load new instance
        model2 = FillProbabilityModel("test_model.pkl")
        self.assertIsNotNone(model2.model)
        
        # Cleanup
        if os.path.exists("test_model.pkl"):
            os.remove("test_model.pkl")

if __name__ == "__main__":
    unittest.main()
