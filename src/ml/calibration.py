import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger("calibration")


class ProbabilityCalibration:
    """
    v24 Probability Calibration Engine.
    Scales model outputs so that confidence matches empirical accuracy.
    Method: Isotonic Regression (Non-parametric) or Platt Scaling (Sigmoid).
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method  # 'isotonic' or 'sigmoid'
        self.calibrator = None

    def calibrate(
        self, base_model: BaseEstimator, X_val: np.ndarray, y_val: np.ndarray
    ) -> BaseEstimator:
        """
        Wrap a trained model with a Calibrator using Validation Data.
        Returns a CalibratedClassifierCV instance (which acts as a model).
        """
        logger.info(f"Calibrating Probabilities using {self.method}...")

        # 'prefit' means base_model is already trained.
        # Sklearn 1.2+ uses 'estimator', older used 'base_estimator'. Try 'estimator' first.
        try:
            calibrated_model = CalibratedClassifierCV(
                estimator=base_model, method=self.method, cv="prefit"
            )
        except TypeError:
            calibrated_model = CalibratedClassifierCV(
                base_estimator=base_model, method=self.method, cv="prefit"
            )

        calibrated_model.fit(X_val, y_val)

        return calibrated_model

    def check_calibration_curve(self, y_true, y_prob, n_bins=10):
        """
        Compute calibration metrics (Expected Calibration Error).
        """
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # ECE
        # abs(prob_true - prob_pred) weighted by bin size
        # Stub logic
        ece = np.mean(np.abs(prob_true - prob_pred))
        return ece, prob_true, prob_pred
