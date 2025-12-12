import logging
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger("ml_analysis")


class WalkForwardCV:
    """
    Time-series strict Walk-Forward Cross-Validation.
    Prevents look-ahead bias by expanding training window and moving test window.
    """

    def __init__(
        self, n_splits: int = 5, train_period: int = 1000, test_period: int = 200
    ):
        self.n_splits = n_splits
        self.train_period = train_period
        self.test_period = test_period

    def split(
        self, data: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Custom generator for walk-forward splits.
        Yields (train_df, test_df)
        """
        n = len(data)
        # Check if data is sufficient
        min_required = self.train_period + self.test_period
        if n < min_required:
            logger.warning(f"Data length {n} < required {min_required} for WF-CV.")
            return

        # Simple expanding window or sliding window?
        # Institutional standard: Expanding Training, Sliding Test.

        # Start index of the first test set
        for i in range(self.n_splits):
            # We want n_splits test blocks at the end? Or iterate through time?
            # Let's iterate from back? Or forward?
            # Standard:
            # Iter 1: Train [0...T], Test [T...T+k]
            # Iter 2: Train [0...T+k], Test [T+k...T+2k]

            # Dynamic calculation to fit n_splits
            # Available test samples = n - train_start_min
            # But let's stick to the parameters provided.

            # Start of test window
            test_start = self.train_period + i * self.test_period
            test_end = test_start + self.test_period

            if test_end > n:
                break

            train_idx = range(0, test_start)
            test_idx = range(test_start, test_end)

            yield data.iloc[train_idx], data.iloc[test_idx]


class FeatureImportanceAnalysis:
    """
    SHAP-based Feature Attribution analysis.
    """

    @staticmethod
    def calculate_shap_values(
        model, X_sample: np.ndarray, feature_names: List[str] = None
    ):
        """
        Calculate SHAP values for a given model and sample.
        Handles XGBoost, LightGBM, CatBoost directly.
        """
        logger.info("📊 Calculating SHAP values...")

        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)

            # Summary plot (usually just return values, plotting is UI/Notebook side)
            # shap.summary_plot(shap_values, X_sample)

            return shap_values
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return None

    @staticmethod
    def get_top_features(
        shap_values, feature_names: List[str], top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top N most important features by mean absolute SHAP value.
        """
        if isinstance(shap_values, shap.Explanation):
            vals = np.abs(shap_values.values).mean(0)
        else:
            vals = np.abs(shap_values).mean(0)

        feature_importance = pd.DataFrame(
            list(zip(feature_names, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )

        return list(feature_importance.head(top_n).itertuples(index=False, name=None))
