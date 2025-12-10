
import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path

logger = logging.getLogger("loss_guard")

class LossPredictionModel:
    """
    Predicts the probability of a trade resulting in a loss (Stop Loss hit or negative return).
    Used as a Veto/Safety mechanism.
    Feature Set: [ret_1h, ret_4h, vol_1h, skew, funding_flip, spread_regime]
    """
    def __init__(self, model_path="data/models/loss_guard.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.features = ["ret_1h", "ret_4h", "vol_1h", "skew", "funding_flip", "spread_regime"]
        self.threshold = 0.6
        self.load()

    def load(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ LossGuard Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load LossGuard Model: {e}")
                self.model = None
        else:
            logger.warning(f"⚠️ LossGuard Model not found at {self.model_path}. Using Dummy Mode (Always Safe).")
            self.model = None

    def predict(self, feature_dict):
        """
        Returns probability of LOSS (0.0 to 1.0).
        If model is missing, returns 0.0 (Safe).
        """
        if self.model is None:
            return 0.0

        # Construct vector
        try:
            X = pd.DataFrame([feature_dict])[self.features]
            # XGBoost/Sklearn predict_proba
            prob_loss = self.model.predict_proba(X)[0, 1]
            return float(prob_loss)
        except Exception as e:
            logger.error(f"LossGuard Prediction Error: {e}")
            return 0.0

    def train_dummy(self):
        """
        Train a dummy model for testing/initialization.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            
            # Synthetic Data
            X = pd.DataFrame(np.random.randn(100, 6), columns=self.features)
            # Make 'vol_1h' correlated with loss
            y = (X["vol_1h"] > 1.0).astype(int) 
            
            clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
            clf.fit(X, y)
            
            self.model = clf
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, self.model_path)
            logger.info("✅ Dummy LossGuard Model trained and saved.")
        except ImportError:
            logger.error("sklearn not found. Cannot train dummy model.")

    def check_veto(self, feature_dict):
        """
        Returns True if trade should be BLOCKED.
        """
        prob = self.predict(feature_dict)
        if prob > self.threshold:
            logger.warning(f"🛡️ LossGuard VETO: P(Loss) {prob:.2f} > {self.threshold}")
            return True, prob
        return False, prob
