from collections import deque
from typing import List

import numpy as np


class AnomalyDetector:
    """
    Detects anomalies in streaming metric data using Z-Score.
    """

    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window = deque(maxlen=window_size)
        self.threshold = threshold

    def update(self, value: float) -> bool:
        """
        Add new value and check if it is an anomaly.
        """
        if len(self.window) < 10:
            self.window.append(value)
            return False

        data = np.array(self.window)
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return False

        z_score = (value - mean) / std

        self.window.append(value)

        if abs(z_score) > self.threshold:
            return True  # Anomaly Detected

        return False
