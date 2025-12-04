import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("meta_model")

class MetaModel:
    """
    Stacked Ensemble Model (Meta-Learner).
    Combines predictions from base models (XGBoost, LightGBM, etc.) using Logistic Regression.
    Includes Probability Calibration.
    """
    def __init__(self):
        self.meta_learner = LogisticRegression(solver='lbfgs', C=1.0)
        self.calibrated_model = None
        self.is_trained = False
        
    def train(self, X_base_preds: pd.DataFrame, y: pd.Series):
        """
        Train the meta-learner on base model predictions.
        
        Args:
            X_base_preds: DataFrame where each column is a base model's prediction (probability).
            y: Target labels (0 or 1).
        """
        logger.info(f"🧠 Training Meta-Learner on {len(X_base_preds)} samples...")
        
        # Use CalibratedClassifierCV for probability calibration
        self.calibrated_model = CalibratedClassifierCV(self.meta_learner, method='sigmoid', cv=5)
        self.calibrated_model.fit(X_base_preds, y)
        
        self.is_trained = True
        logger.info("✅ Meta-Learner Trained & Calibrated.")
        
        # Log coefficients (Average across folds)
        try:
            coefs = np.mean([clf.estimator.coef_ for clf in self.calibrated_model.calibrated_classifiers_], axis=0)
            logger.info(f"Meta Coefficients (Avg): {coefs}")
        except Exception as e:
            logger.warning(f"Could not log coefficients: {e}")

    def predict_proba(self, X_base_preds: pd.DataFrame) -> np.ndarray:
        """
        Generate final calibrated probabilities.
        """
        if not self.is_trained:
            raise ValueError("MetaModel is not trained yet.")
            
        return self.calibrated_model.predict_proba(X_base_preds)[:, 1]

    def save(self, path: Path):
        joblib.dump(self, path)
        logger.info(f"💾 MetaModel saved to {path}")

    @staticmethod
    def load(path: Path):
        return joblib.load(path)

def generate_mock_data(n_samples=1000):
    """Generate synthetic base predictions and targets for testing."""
    np.random.seed(42)
    y = np.random.randint(0, 2, n_samples)
    
    # Simulate decent base models (correlated with y)
    pred_xgb = np.clip(y * 0.7 + np.random.normal(0, 0.2, n_samples), 0, 1)
    pred_lgb = np.clip(y * 0.65 + np.random.normal(0, 0.25, n_samples), 0, 1)
    pred_lstm = np.clip(y * 0.6 + np.random.normal(0, 0.3, n_samples), 0, 1)
    
    X = pd.DataFrame({
        'xgb': pred_xgb,
        'lgb': pred_lgb,
        'lstm': pred_lstm
    })
    return X, y

if __name__ == "__main__":
    # 1. Generate Data
    X, y = generate_mock_data()
    
    # 2. Split
    split = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split], y[:split]
    X_test, y_test = X.iloc[split:], y[split:]
    
    # 3. Train Meta Model
    meta = MetaModel()
    meta.train(X_train, y_train)
    
    # 4. Predict
    final_probs = meta.predict_proba(X_test)
    
    # 5. Evaluate
    from sklearn.metrics import roc_auc_score, log_loss
    auc = roc_auc_score(y_test, final_probs)
    loss = log_loss(y_test, final_probs)
    
    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Test LogLoss: {loss:.4f}")
    
    # 6. Save
    meta.save(Path("meta_model_test.pkl"))
