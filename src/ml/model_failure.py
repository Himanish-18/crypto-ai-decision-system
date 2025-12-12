from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class ModelFailureDetector:
    """
    v19 Model Failure Detector.
    Predicts P(Failure) given Market Context.
    Failure = Signal was generated but resulted in Loss.

    Features: Volatility, Spread, Time of Day, Ensemble Variance (Disagreement).
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )

    def fit(self, X_context: pd.DataFrame, y_failure: pd.Series):
        """
        X_context: Context features (Vol, Spread...)
        y_failure: 1 if trade failed, 0 if trade succeeded.
        """
        self.model.fit(X_context, y_failure)

    def predict_failure_prob(self, context: Dict[str, float]) -> float:
        # Convert dict to DataFrame
        df = pd.DataFrame([context])

        # Ensure col order match if trained
        # (Simplified: assume user passes correct dict keys. In prod, store feature_names_in_)

        try:
            return self.model.predict_proba(df)[0, 1]
        except Exception:
            # Cold start or mismatched features
            return 0.5

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
