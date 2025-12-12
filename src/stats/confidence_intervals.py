from typing import Tuple

import numpy as np


class ConfidenceIntervals:
    """
    Confidence Intervals for Financial Metrics.
    """

    @staticmethod
    def sharpe_ratio_ci(
        sharpe: float, n_samples: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Analytic CI for Sharpe Ratio (Lo, 2002).
        SE = sqrt((1 + 0.5 * Sharpe^2) / N)
        """
        se = np.sqrt((1 + 0.5 * sharpe**2) / n_samples)
        z = 1.96  # Approx for 95%
        return (sharpe - z * se, sharpe + z * se)
