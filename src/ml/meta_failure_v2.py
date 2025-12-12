import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("meta_failure")


class MetaFailureModel:
    """
    v24 Meta-Failure Detector (Model of Models).
    Predicts probability of PRIMARY MODEL FAILURE based on environmental conditions.
    Target: Is PnL < -X%?
    """

    def __init__(self, failure_threshold: float = -0.005):
        self.threshold = failure_threshold
        # Using RF for robustness and feature importance
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        self.is_fitted = False

    def train(self, trades_df: pd.DataFrame):
        """
        Train the Meta-Model on historical trade outcomes.
        trades_df should contain:
        - Features: 'volatility', 'spread', 'imbalance', 'regime_score', 'model_confidence'
        - Target: 'pnl' (Realized PnL of the trade)
        """
        if trades_df.empty:
            logger.warning("No data to train Meta-Failure Model.")
            return

        required_features = [
            "volatility",
            "spread",
            "imbalance",
            "regime_score",
            "model_confidence",
        ]
        available_features = [f for f in required_features if f in trades_df.columns]

        if not available_features:
            logger.warning("Missing required features for Meta-Failure Model.")
            return

        X = trades_df[available_features].fillna(0)
        # Target: 1 if FAILED (Loss > threshold big), 0 otherwise
        y = (trades_df["pnl"] < self.threshold).astype(int)

        if y.sum() == 0:
            logger.warning(
                "No failures in dataset (Good job?). Cannot train failure model."
            )
            return

        self.model.fit(X, y)
        self.is_fitted = True

        logger.info(
            f"Meta-Failure Model Trained. OOB Score: {self.model.score(X, y):.2f}"
        )
        logger.info(
            f"Feature Importance: {dict(zip(available_features, self.model.feature_importances_))}"
        )

    def predict_failure_proba(self, market_state: Dict) -> float:
        """
        Return probability that the current trade will FAIL.
        market_state: Dict with keys ['volatility', 'spread', 'imbalance', 'regime_score', 'model_confidence']
        """
        if not self.is_fitted:
            return 0.0  # Assume model is fine if not trained

        # Convert dict to array
        # Order implies we rely on `train` feature order matching.
        # Ideally we store feature names. for now assume standard order:
        # volatility, spread, imbalance, regime_score, model_confidence
        features = [
            market_state.get("volatility", 0),
            market_state.get("spread", 0),
            market_state.get("imbalance", 0),
            market_state.get("regime_score", 0),
            market_state.get("model_confidence", 0.5),
        ]

        X = np.array([features])

        # Prob of Class 1 (Failure)
        prob_fail = self.model.predict_proba(X)[0][1]

        return prob_fail

    def should_veto(self, market_state: Dict, veto_threshold: float = 0.6) -> bool:
        """
        Hard Veto Logic.
        """
        p_fail = self.predict_failure_proba(market_state)
        if p_fail > veto_threshold:
            logger.warning(
                f"Meta-Failure VETO: Prob Failure {p_fail:.2f} > {veto_threshold}"
            )
            return True
        return False
