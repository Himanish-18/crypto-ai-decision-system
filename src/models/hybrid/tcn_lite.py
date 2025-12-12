import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger("tcn_lite")


class TCNLiteProxy:
    def __init__(self):
        # Histogram-based Gradient Boosting is fast and handles large datasets/interactions well
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_depth=5,
            random_state=42,
            early_stopping=True,
        )
        self.is_fitted = False
        self.feature_cols = []

    def fit(self, df: pd.DataFrame, target_col="target"):
        logger.info("🌊 Training TCN-Lite Proxy (GBM)...")

        # Select Features (Automated)
        exclude = ["timestamp", "regime", "rl_signal", target_col]

        # 1. Filter numeric only
        candidates = [
            c
            for c in df.columns
            if c not in exclude and np.issubdtype(df[c].dtype, np.number)
        ]

        # 2. Strict Filter: Remove Targets and Future Leaks
        # Remove anything with 'fwd', 'target', 'y_', 'future'
        candidates = [
            c
            for c in candidates
            if not any(x in c for x in ["fwd", "target", "y_", "future"])
        ]

        # 3. Robustness Filter: Remove 'alpha_' columns if they are prone to missing in live
        # (Based on deployment logs showing key errors for alpha_whale_flow etc.)
        candidates = [c for c in candidates if not c.startswith("alpha_")]

        self.feature_cols = candidates

        X = df[self.feature_cols]
        y = df[target_col]

        self.model.fit(X, y)
        self.is_fitted = True

        score = self.model.score(X, y)
        logger.info(f"✅ TCN-Lite Training Accuracy: {score:.4f}")

    def predict_trend(self, row_data: pd.DataFrame) -> float:
        """
        Predict trend prob for a single row (latest).
        """
        if not self.is_fitted:
            return 0.5

        # Ensure input has features
        # HistGBM handles NaNs natively, but we need to ensure columns exist!
        # Reindex checks for missing columns and adds them as NaN (or fill_value)
        X = row_data.reindex(columns=self.feature_cols, fill_value=0.0)

        prob = self.model.predict_proba(X)[0, 1]
        return prob

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"💾 TCN-Lite saved to {path}")

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
