import logging

import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger("regime_detection")


class MarketRegimeDetector:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.hmm_model = GaussianHMM(
            n_components=n_components, covariance_type="diag", n_iter=100
        )
        self.garch_model = None
        self.regime_map = {}  # Map hidden states to 'Bull', 'Bear', 'Chop'

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM: Log Returns and Range.
        """
        df = df.copy()
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["range"] = (df["high"] - df["low"]) / df["close"]
        df.dropna(inplace=True)

        X = df[["log_ret", "range"]].values
        return X

    def fit_hmm(self, df: pd.DataFrame):
        """
        Fit HMM model to historical data.
        """
        X = self.prepare_features(df)
        self.hmm_model.fit(X)

        # Heuristic to label regimes
        means = self.hmm_model.means_
        # Sort by volatility (range)
        sorted_indices = np.argsort(means[:, 1])

        # 0: Low Vol (Chop), 1: Med Vol, 2: High Vol
        # We need to distinguish Bull/Bear by returns (means[:, 0])

        logger.info(f"HMM Means: {means}")
        logger.info("HMM Fitted successfully")

    def predict_regime(self, df: pd.DataFrame) -> int:
        """
        Predict current regime.
        """
        X = self.prepare_features(df)
        hidden_states = self.hmm_model.predict(X)
        return hidden_states[-1]

    def fit_garch(self, returns: pd.Series):
        """
        Fit GARCH(1,1) model for volatility.
        """
        model = arch_model(returns * 100, vol="Garch", p=1, q=1)
        self.garch_model = model.fit(disp="off")
        logger.info(self.garch_model.summary())

    def predict_volatility(self, horizon: int = 1) -> float:
        """
        Predict volatility for next N steps.
        """
        if self.garch_model is None:
            return 0.0
        forecast = self.garch_model.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1, :][0])
