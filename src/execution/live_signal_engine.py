import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("live_signal_engine")

class LiveSignalEngine:
    def __init__(self, model_path: Path, scaler_path: Path):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.regime_detector = None # Initialize regime_detector
        self.load_artifacts()

    def load_artifacts(self):
        """Load model and scaler."""
        logger.info(f"📥 Loading model from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            
        logger.info(f"📥 Loading scaler from {self.scaler_path}...")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # Load Regime Detector
        self.regime_model_path = self.model_path.parent / "regime_model.pkl"
        if self.regime_model_path.exists():
            logger.info(f"📥 Loading regime detector from {self.regime_model_path}...")
            with open(self.regime_model_path, "rb") as f:
                self.regime_detector = pickle.load(f)
        else:
            logger.warning("⚠️ Regime detector not found. Using default thresholds.")
            self.regime_detector = None

    def process_candle(self, candle_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process a single candle (row) to generate a signal.
        candle_data: DataFrame with 1 row containing all features.
        """
        # Prepare Features
        # We need to ensure columns match what the model expects
        # The model expects 101 features.
        # We assume candle_data has all of them.
        
        # Drop non-feature cols if present
        exclude_cols = ["timestamp", "y_direction_up", "btc_ret_fwd_1", "y_pred", "y_prob", "signal_prob", "atr_pct", "is_shock", "is_uptrend", "signal_consistent", "entry_signal"]
        feature_cols = [c for c in candle_data.columns if c not in exclude_cols]
        
        # Check feature count
        if hasattr(self.model, "n_features_in_"):
            expected = self.model.n_features_in_
            current = len(feature_cols)
            if current != expected:
                # Try to align
                # This is tricky in live mode. Ideally we pass exact feature list.
                # For now, let's assume features are correct or use model's feature_names_in_ if available
                if hasattr(self.model, "feature_names_in_"):
                    feature_cols = self.model.feature_names_in_
        
        X = candle_data[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Predict Probability
        prob = self.model.predict_proba(X_scaled)[0, 1]
        
        # --- Regime Detection & Dynamic Thresholds ---
        regime = "Unknown"
        threshold = 0.53 # Default
        
        if self.regime_detector:
            try:
                regime = self.regime_detector.predict(candle_data)
                
                if regime == "Bull":
                    threshold = 0.51 # Aggressive
                elif regime == "Bear":
                    threshold = 0.60 # Conservative
                elif regime == "Sideways":
                    threshold = 0.65 # Very Conservative
                    
            except Exception as e:
                logger.error(f"Regime detection failed: {e}")
        
        # Strategy Logic
        # 1. Probability Threshold
        prob_signal = 1 if prob > threshold else 0
        
        # 2. Trend Filter (RSI > 50 OR MACD > 0)
        # In Bear regime, maybe we want stricter trend filter?
        # For now, keep it simple: Trend Filter always ON
        rsi = candle_data["btc_rsi_14"].iloc[0]
        macd = candle_data["btc_macd"].iloc[0]
        is_uptrend = 1 if (rsi > 50 or macd > 0) else 0
        
        # Final Signal
        signal = 1 if (prob_signal == 1 and is_uptrend == 1) else 0
        
        # Context
        context = {
            "rsi": rsi,
            "macd": macd,
            "zscore": candle_data["btc_zscore_5"].iloc[0],
            "atr": candle_data["btc_atr_14"].iloc[0],
            "sentiment_shock": candle_data.get("sentiment_shock", pd.Series([0])).iloc[0],
            "atr_pct": candle_data.get("atr_pct", pd.Series([0])).iloc[0],
            "regime": regime,
            "threshold": threshold
        }
        
        return {
            "timestamp": candle_data["timestamp"].iloc[0],
            "prediction_prob": prob,
            "signal": signal,
            "signal_confidence": abs(prob - 0.5) * 2, # 0 to 1 scale
            "strategy_context": context
        }
