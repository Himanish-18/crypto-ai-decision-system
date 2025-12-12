
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("fill_prob_model")

# Try importing XGBoost, fallback to RandomForest
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import RandomForestClassifier

class FillProbabilityModel:
    """
    v43 ML model to estimate P(Fill | MarketState).
    Predicts probability of a passive order being filled within T seconds without adverse selection.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self.feature_cols = [
            "spread_bps", "depth_skew", "volatility_1m", 
            "trade_flow_imbalance", "dist_to_mid"
        ]
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
        else:
            self._init_model()

    def _init_model(self):
        """Initialize a fresh model."""
        if HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.05, 
                objective='binary:logistic'
            )
            logger.info("Initialized XGBoost for Fill Probability.")
        else:
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5)
            logger.info("Initialized RandomForest for Fill Probability (XGB not found).")
            
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model on historical execution data.
        X: Feature DataFrame
        y: Binary target (1 = Filled Good, 0 = Unfilled/Adverse)
        """
        if self.model is None:
            self._init_model()
            
        logger.info(f"Training FillProb Model on {len(X)} samples...")
        self.model.fit(X[self.feature_cols], y)
        logger.info("Training complete.")

    def predict_fill_prob(self, market_state: Dict[str, float]) -> float:
        """
        Predict P(Fill) for a single state vector.
        """
        if self.model is None:
            return 0.5 # Default uncertainty
            
        # Convert state dict to DF
        try:
            row = pd.DataFrame([market_state], columns=self.feature_cols)
            # Handle missing feat
            row = row.fillna(0.0)
            
            if HAS_XGB:
                prob = self.model.predict_proba(row)[0][1]
            else:
                prob = self.model.predict_proba(row)[0][1]
                
            return float(prob)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {path}")
