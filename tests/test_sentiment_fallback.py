import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from src.features.sentiment_features import SentimentFeatures


class TestSentimentFallback(unittest.TestCase):
    def setUp(self):
        # Create Dummy OHLCV
        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        self.df = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.normal(100, 5, 100),
                "high": np.random.normal(105, 5, 100),
                "low": np.random.normal(95, 5, 100),
                "close": np.random.normal(100, 5, 100),
                "volume": np.random.normal(1000, 100, 100),
            }
        )

    def test_synthetic_fallback_trigger(self):
        """Test that synthetic logic is triggered when fundingRate is missing."""
        # Ensure 'fundingRate' is NOT in df

        df_out = SentimentFeatures.calculate_proxies(self.df)

        # Check if synthetic column exists
        self.assertIn("feat_synthetic_sentiment", df_out.columns)
        # Check if main proxy uses synthetic
        # Should be equal (except for fillna differences maybe)
        self.assertTrue(
            np.allclose(
                df_out["feat_synthetic_sentiment"].fillna(0),
                df_out["feat_sentiment_proxy"].fillna(0),
            )
        )

        # Check score range
        score = df_out["feat_synthetic_sentiment"].mean()
        self.assertTrue(-1.0 <= score <= 1.0)

    def test_normal_logic(self):
        """Test that normal logic runs when fundingRate is present."""
        df_normal = self.df.copy()
        df_normal["fundingRate"] = 0.0001
        df_normal["openInterest"] = 100000

        df_out = SentimentFeatures.calculate_proxies(df_normal)

        # Should NOT trigger synthetic (strictly speaking not blocked, but standard flow preferred)
        # Actually our implementation calls synthetic ONLY if funding is missing.
        self.assertNotIn("feat_synthetic_sentiment", df_out.columns)
        self.assertIn("feat_sentiment_proxy", df_out.columns)


if __name__ == "__main__":
    unittest.main()
