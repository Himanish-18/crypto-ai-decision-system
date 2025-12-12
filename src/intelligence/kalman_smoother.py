import numpy as np


class KalmanSmoother:
    """
    1D Kalman Filter for smoothing probability scores.
    """

    def __init__(self, process_noise: float = 0.005, measurement_noise: float = 0.03):
        self.q = process_noise  # Process variance
        self.r = measurement_noise  # Measurement variance
        self.p = 1.0  # Estimation error covariance
        self.x = 0.5  # Initial state (Probability 0.5)

    def smooth(self, measurement: float) -> float:
        """
        Update state with new measurement and return smoothed value.
        """
        # Prediction Step
        # x_pred = x_prev (Assume constant state model)
        # p_pred = p_prev + q
        p_pred = self.p + self.q

        # Update Step
        # Kalman Gain: k = p_pred / (p_pred + r)
        k = p_pred / (p_pred + self.r)

        # State Update: x = x_pred + k(measurement - x_pred)
        self.x = self.x + k * (measurement - self.x)

        # Covariance Update: p = (1 - k) * p_pred
        self.p = (1 - k) * p_pred

        return float(self.x)
