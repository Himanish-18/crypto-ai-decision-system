import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Remove global mocks. We will patch in setUp.
# The original code likely imported modules that used these mocks immediately.
# We need to ensure we patch BEFORE import if we want to mock deps.

import os

sys.path.append(os.getcwd())

from src.models.loss_prediction_model import LossPredictionModel


class TestLossGuard(unittest.TestCase):
    def test_model_veto(self):
        """Test the logic of check_veto."""
        model = LossPredictionModel()
        model.model = MagicMock()
        # Mock predict_proba to return [P(Safe), P(Loss)]
        model.model.predict_proba.return_value = np.array([[0.1, 0.9]])

        # FIX: Provide ALL required features
        feat = {
            "ret_1h": 0.01,
            "ret_4h": 0.05,
            "vol_1h": 0.02,
            "skew": 0.1,
            "funding_flip": 0,
            "spread_regime": 0,
        }
        veto, prob = model.check_veto(feat)

        self.assertTrue(veto)
        self.assertEqual(prob, 0.9)

        # Test Safe
        model.model.predict_proba.return_value = np.array([[0.8, 0.2]])
        veto, prob = model.check_veto(feat)
        self.assertFalse(veto)
        self.assertEqual(prob, 0.2)


if __name__ == "__main__":
    unittest.main()
