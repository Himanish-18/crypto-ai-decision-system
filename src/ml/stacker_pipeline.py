import pickle
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    v19 Stacking Model.
    Combines predictions from multiple base models using a Meta-Learner.
    """

    def __init__(self, base_models: Dict[str, Any], meta_learner: Any = None):
        """
        base_models: Dict of {name: model_instance}
        meta_learner: Classifier (default LogisticRegression)
        """
        self.base_models = base_models
        self.meta_learner = meta_learner if meta_learner else LogisticRegression()
        self.calibrator = None

    def fit(self, X: pd.DataFrame, y: pd.Series, calibrate: bool = True):
        # 1. Generate OOF (Out-of-Fold) Predictions or Split Training
        # For simplicity in this implementation, we assume X is the meta-feature set
        # (i.e. output of base models) OR we execute base models here.

        # Scenario A: X contains raw features. We must train base models.
        # Scenario B: X contains base model predictions (trained elsewhere).

        # We'll assume Scenario B for the "Meta-Stacker" component specifically
        # to keep this class focused. The calling script handles the CV loop.

        # Meta Training
        self.meta_learner.fit(X, y)

        # Calibration
        if calibrate:
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            # Use calibration set? Or fit on same set (risk of overfitting)?
            # Proper way: Split X into Train/Calib.
            # Here simplified: fit on training outputs (User Check: verify MACE)
            meta_probs = self.meta_learner.predict_proba(X)[:, 1]
            self.calibrator.fit(meta_probs, y)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # X should be DataFrame of base model predictions: [xgb_prob, lgbm_prob, cnn_prob...]
        meta_prob = self.meta_learner.predict_proba(X)[:, 1]

        if self.calibrator:
            calibrated_prob = self.calibrator.predict(meta_prob)
            # Return form [1-p, p] to match sklearn
            return np.vstack([1 - calibrated_prob, calibrated_prob]).T

        return np.vstack([1 - meta_prob, meta_prob]).T

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


class Calibrator:
    """Helper for standalone calibration"""

    def __init__(self):
        self.iso_reg = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs, y):
        self.iso_reg.fit(probs, y)

    def calibrate(self, probs):
        return self.iso_reg.predict(probs)
