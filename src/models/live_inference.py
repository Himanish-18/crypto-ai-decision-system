import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("live_inference")


class LiveInferenceEngine:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.regime_model = None
        self.ensemble_model = None
        self.load_models()

    def load_models(self):
        try:
            self.regime_model = joblib.load(self.models_dir / "regime_model.pkl")
            self.ensemble_model = joblib.load(self.models_dir / "ensemble_model.pkl")
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Generate prediction based on features and regime.
        """
        if not self.regime_model or not self.ensemble_model:
            return {"probability": 0.5, "direction": 0, "confidence": 0.0}

        # Prepare feature vector (must match training columns)
        # Training features: ['order_imbalance_10', 'cvd_1h', 'vwap_deviation', 'rsi', 'regime']
        # Mapped from live features:
        # order_imbalance_10 -> ofi (proxy)
        # cvd_1h -> cvd_10s (scaled proxy)
        # vwap_deviation -> (mid - micro) / mid (proxy)
        # rsi -> 50 (neutral placeholder if not computed)

        # Note: In a real system, we need exact feature parity.
        # Here we map available live features to model inputs.

        # Mock Regime Prediction (needs history, here we use placeholder or simple logic)
        # For simplicity in this live tick loop, we assume a regime or use a simple heuristic
        # Real implementation would maintain a history buffer to feed HMM.
        current_regime = 1  # Assume 'Trend' for now

        input_features = pd.DataFrame(
            [
                {
                    "order_imbalance_10": features.get(
                        "ofi", 0
                    ),  # OFI is similar concept
                    "cvd_1h": features.get("cvd_10s", 0),
                    "vwap_deviation": (
                        features.get("mid_price", 0) - features.get("microprice", 0)
                    )
                    / features.get("mid_price", 1),
                    "rsi": 50.0,  # Placeholder
                    "regime": current_regime,
                }
            ]
        )

        prob = self.ensemble_model.predict_proba(input_features)[0]

        # Regime Logic
        # If Mean Reversion (Regime 0), fade the signal or use different threshold
        # If Trend (Regime 1), follow signal

        direction = 1 if prob > 0.6 else (-1 if prob < 0.4 else 0)
        confidence = abs(prob - 0.5) * 2

        return {
            "probability": prob,
            "direction": direction,
            "confidence": confidence,
            "regime": current_regime,
        }
