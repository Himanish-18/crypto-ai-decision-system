
import unittest
import numpy as np
from collections import deque
import sys
import os

# Mock dependencies not needed for this isolated test
import os
sys.path.append(os.getcwd())

from src.risk_engine.correlation_guard import CorrelationGuard

class TestCorrelationGuard(unittest.TestCase):
    def test_perfect_correlation(self):
        """Test reaction to perfectly correlated assets."""
        guard = CorrelationGuard(window_size=10)
        
        # Feed identical movements
        base_price = 100
        for i in range(15):
            noise = i * 1.0 # Trend up
            guard.update({
                "BTC": base_price + noise,
                "ETH": (base_price + noise) * 0.1,
                "SOL": (base_price + noise) * 0.01,
                "LTC": (base_price + noise) * 0.02
            })
            
        corr_matrix = guard.compute_matrix()
        # Verify matrix exists
        self.assertIsNotNone(corr_matrix)
        
        # Verify high correlation
        # Diagonals are 1.0, off-diagonals should be 1.0 too
        self.assertGreater(corr_matrix.iloc[0, 1], 0.99)
        
        # Verify Risk Modifiers
        scalar, hedge, debug = guard.calculate_risk_modifiers(current_pos_size=1.0)
        
        # Max Corr ~ 1.0 > 0.6
        # Expected reduction: (1.0 - 0.6) * 1.66 = 0.66 reduction -> Min capped at 0.5?
        # Logic: max(0.5, 1.0 - reduction)
        # 1.0 - 0.66 = 0.34 -> Clamped to 0.5
        self.assertAlmostEqual(scalar, 0.5, places=1)
        self.assertTrue(hedge) # Pos size 1.0 > 0.5 & Corr > 0.7

    def test_noise_correlation(self):
        """Test reaction to uncorrelated noise."""
        guard = CorrelationGuard(window_size=20)
        
        np.random.seed(42)
        for i in range(25):
            guard.update({
                "BTC": 100 + np.random.normal(),
                "ETH": 50 + np.random.normal(), # Independent
                "SOL": 20 + np.random.normal(),
                "LTC": 10 + np.random.normal()
            })
            
        scalar, hedge, debug = guard.calculate_risk_modifiers(current_pos_size=1.0)
        
        # Correlation should be low
        print(f"DEBUG Noise Check: Max Corr {debug['max_corr']}")
        
        self.assertEqual(scalar, 1.0, "Should not reduce size for noise")
        self.assertFalse(hedge, "Should not hedge for noise")

if __name__ == "__main__":
    unittest.main()
