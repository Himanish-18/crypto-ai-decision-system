import logging
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None  # Handle missing dependency

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger("regime_ensemble")


class RegimeEnsemble:
    """
    v21 Regime Detection Ensemble.
    Combines Unsupervised (HMM) and Supervised (XGB) models to predict Market Regime Risk.

    Regimes:
    0: Low Vol / Bull (Safe)
    1: High Vol / Bear (Risk)
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

        if GaussianHMM:
            self.hmm = GaussianHMM(
                n_components=n_components, covariance_type="full", n_iter=100
            )
        else:
            self.hmm = None
            logger.warning("hmmlearn not installed. HMM disabled.")

        self.xgb_model = None  # XGBClassifier or Booster
        self.risk_threshold = 0.5

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Train models.
        X: Features (Returns, Volatility, funding, etc.)
        y: Labels (1=Crash/Risk, 0=Normal) - Only for XGB.
        """
        # 1. Train HMM (Unsupervised)
        if self.hmm:
            # HMM expects continuous features (Returns, Vol).
            # We assume X columns are properly scaled.
            self.hmm.fit(X)
            logger.info("HMM Trained.")

        # 2. Train XGB (Supervised)
        if xgb and y is not None:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1
            )
            self.xgb_model.fit(X, y)
            logger.info("XGB Regime Model Trained.")

    def predict_risk(
        self, feature_vector: pd.DataFrame
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Predict Regime Risk Score [0, 1].
        """
        avg_risk = 0.0
        details = {}
        count = 0

        # HMM Score (Latent State)
        if self.hmm:
            try:
                # Predict latent state
                state = self.hmm.predict(feature_vector)[0]
                # We need to map State -> Risk.
                # Heuristic: The state with higher variance in 'volatility' feature is Risk.
                # Use pre-calibrated map or simple heuristic if trained:
                # Let's assume we map State 1 to Risk if we forced 2 states.
                # For robustness, we check the posterior prob of the "noisy" state.
                probs = self.hmm.predict_proba(feature_vector)[0]

                # Assume State 1 is High Vol (Need calibration in real training)
                # Here we just treat prob[1] as risk for demo
                hmm_risk = probs[1]
                avg_risk += hmm_risk
                count += 1
                details["hmm_state"] = int(state)
                details["hmm_prob"] = hmm_risk
            except Exception as e:
                logger.error(f"HMM Predict Error: {e}")

        # XGB Score
        if self.xgb_model:
            try:
                xgb_prob = self.xgb_model.predict_proba(feature_vector)[0, 1]
                avg_risk += xgb_prob
                count += 1
                details["xgb_prob"] = xgb_prob
            except Exception as e:
                logger.error(f"XGB Predict Error: {e}")

        if count > 0:
            final_risk = avg_risk / count
        else:
            final_risk = 0.0

        return final_risk, details

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
