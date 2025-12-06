import pandas as pd
import numpy as np
import pickle
import json
import logging
from pathlib import Path

logger = logging.getLogger("rl_signals")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODEL_PATH = MODELS_DIR / "rl_agent_rf.pkl"
FEATURES_PATH = MODELS_DIR / "rl_features.json"

class RLSignalEngine:
    def __init__(self):
        self.model = None
        self.feature_cols = []
        self._load_model()
        
    def _load_model(self):
        try:
            if not MODEL_PATH.exists():
                logger.warning(f"⚠️ RL Model not found at {MODEL_PATH}")
                return
                
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
                
            with open(FEATURES_PATH, "r") as f:
                self.feature_cols = json.load(f)
                
            logger.info("✅ RL Agent Loaded Successfully.")
            
        except Exception as e:
            logger.error(f"❌ Failed to load RL Agent: {e}")
            
    def get_signal(self, candle_data: pd.DataFrame) -> int:
        """
        Get RL Action for the latest candle.
        Returns: 1 (Buy/Long), 0 (Hold/Neutral), -1 (Sell/Short - if supported)
        For this simplified proxy, it returns 1 (Buy High Confidence) or 0 (Neutral).
        """
        if self.model is None:
            return 0
            
        try:
            # Ensure we have the latest row
            latest = candle_data.iloc[[-1]].copy()
            
            # Feature alignment
            # Fill missing cols with 0
            for c in self.feature_cols:
                if c not in latest.columns:
                    latest[c] = 0.0
                    
            X = latest[self.feature_cols]
            
            # Predict
            # Model target was 1=Profitable Move, 0=Not Profitable
            pred = self.model.predict(X)[0]
            
            # If pred == 1, it means the agent expects Return > Fee -> BUY
            return int(pred)
            
        except Exception as e:
            logger.error(f"RL Prediction Error: {e}")
            return 0
            
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Process entire dataframe for backtesting"""
        if self.model is None:
            return np.zeros(len(df))
            
        # Align features
        X = df.copy()
        missing = [c for c in self.feature_cols if c not in X.columns]
        for c in missing:
            X[c] = 0.0
            
        X = X[self.feature_cols]
        # Replace inf/nan
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        
        return self.model.predict(X)
