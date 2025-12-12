import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("regime_detector")


class RegimeDetector:
    def __init__(
        self, n_components: int = 3, covariance_type: str = "full", n_iter: int = 100
    ):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Mapping from hidden state index to semantic label (Bull, Bear, Sideways)
        # This needs to be determined after training by analyzing state means
        self.state_map = {}

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features for regime detection.
        We want features that capture Volatility, Trend, and Momentum.
        """
        df = df.copy()

        # Ensure required columns exist
        required = ["btc_close", "btc_atr_14", "btc_rsi_14"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # 1. Volatility: ATR (Normalized by price)
        df["volatility"] = df["btc_atr_14"] / df["btc_close"]

        # 2. Trend: Returns (Log returns)
        if "btc_ret" in df.columns:
            df["returns"] = df["btc_ret"]
        else:
            df["returns"] = np.log(df["btc_close"] / df["btc_close"].shift(1))

        # 3. Momentum: RSI (Scaled to 0-1 or centered)
        df["momentum"] = (df["btc_rsi_14"] - 50) / 100.0  # Center around 0

        # 4. Skewness (Rolling 30)
        df["skew"] = df["btc_close"].rolling(window=30).skew().fillna(0)

        # Drop NaNs
        df = df.dropna()

        # Select features
        X = df[["volatility", "returns", "momentum", "skew"]].values
        return X

    def fit(self, df: pd.DataFrame):
        """Train the HMM model."""
        logger.info("🧠 Training Regime Detector (HMM)...")

        X = self.prepare_features(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit HMM
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Analyze states to create semantic map
        self._map_states(X_scaled)

        logger.info("✅ Regime Detector Trained.")

    def _map_states(self, X_scaled):
        """
        Map hidden states to semantic labels (Crash, Bear, Sideways, Bull).
        Logic:
        - Crash: Very negative returns, High volatility
        - Bear: Negative returns, High volatility
        - Bull: Positive returns, Low/Medium volatility
        - Sideways: Low returns, Low volatility
        """
        means = self.model.means_
        # means shape: (n_components, n_features) -> [volatility, returns, momentum, skew]

        # We can sort states by 'returns' mean
        sorted_indices = np.argsort(means[:, 1])

        # If 4 components:
        if self.n_components == 4:
            self.state_map = {
                sorted_indices[0]: "Crash",
                sorted_indices[1]: "Bear",
                sorted_indices[2]: "Sideways",
                sorted_indices[3]: "Bull",
            }
        else:
            # Fallback to 3 states
            self.state_map = {
                sorted_indices[0]: "Bear",
                sorted_indices[1]: "Sideways",
                sorted_indices[2]: "Bull",
            }

        logger.info(f"State Mapping: {self.state_map}")
        for i in range(self.n_components):
            logger.info(
                f"State {i}: Vol={means[i,0]:.4f}, Ret={means[i,1]:.4f}, Mom={means[i,2]:.4f}, Skew={means[i,3]:.4f} -> {self.state_map[i]}"
            )

    def predict(self, df: pd.DataFrame) -> str:
        """Predict regime for the latest data point."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        # Predict sequence of states
        hidden_states = self.model.predict(X_scaled)

        # Get last state
        last_state = hidden_states[-1]
        return self.state_map.get(last_state, "Unknown")

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
