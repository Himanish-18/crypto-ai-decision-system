import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger("dqn_mini")


class DQNMiniProxy:
    def __init__(self):
        # We model Q(s, a) where a = 1 (Trade).
        # We only need to predict the EXPECTED RETURN of taking the trade.
        # If Expected Return > 0 (or some margin), we Buy.
        # This simplifies DQN to a Value Function approximator.
        self.model = RandomForestRegressor(
            n_estimators=50, max_depth=5, n_jobs=-1, random_state=42
        )
        self.is_fitted = False

    def prepare_state(self, row, mf_score, cnn_score, tcn_score):
        """
        Construct State Vector [MF, CNN, TCN, Vol, RegimeCode]
        """
        # Encode Regime
        regime = row.get("regime", "Normal")
        regime_code = 0.0
        if regime == "Trend":
            regime_code = 1.0
        elif regime == "High Volatility":
            regime_code = 2.0
        elif regime == "Low Liquidity":
            regime_code = 3.0

        vol = row.get("btc_atr_14", 0) / (row.get("btc_close", 1) + 1e-9)

        return np.array([mf_score, cnn_score, tcn_score, regime_code, vol]).reshape(
            1, -1
        )

    def fit(self, df: pd.DataFrame, mf_scores, cnn_scores, tcn_scores, targets_pnl):
        """
        Train to predict PnL (Reward).
        """
        logger.info("🤖 Training DQN-Mini Proxy (Value Net)...")

        # Construct dataset
        states = []
        rewards = []

        for i in range(len(df)):
            row = df.iloc[i]
            # State
            state = self.prepare_state(row, mf_scores[i], cnn_scores[i], tcn_scores[i])
            states.append(state[0])
            rewards.append(targets_pnl[i])

        X = np.array(states)
        y = np.array(rewards)

        self.model.fit(X, y)
        self.is_fitted = True

        score = self.model.score(X, y)  # R2 score
        logger.info(f"✅ DQN-Mini Value Net R2: {score:.4f}")

    def predict_q_value(self, row, mf_score, cnn_score, tcn_score):
        if not self.is_fitted:
            return 0.0

        state = self.prepare_state(row, mf_score, cnn_score, tcn_score)
        q_val = self.model.predict(state)[0]
        return q_val

    def get_action(self, q_value, threshold=0.001):
        # Action 1 (Buy) if Q > cost threshold
        return 1 if q_value > threshold else 0

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"💾 DQN-Mini saved to {path}")

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)
