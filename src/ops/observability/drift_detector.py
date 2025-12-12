from typing import Dict, Optional

import numpy as np
from scipy import stats


class DriftDetector:
    """
    Detects feature distribution drift using PSI (Population Stability Index)
    and KL Divergence.
    """

    @staticmethod
    def calculate_psi(
        expected: np.ndarray, actual: np.ndarray, buckets: int = 10
    ) -> float:
        """
        Calculate PSI between expected (training) and actual (production) distributions.
        """

        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        # Quantize expected (baseline)
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        exp_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
        act_percents = np.histogram(actual, bins=buckets)[0] / len(actual)

        def sub_psi(e_perc, a_perc):
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        psi_total = np.sum(
            [
                sub_psi(expected_p, actual_p)
                for expected_p, actual_p in zip(exp_percents, act_percents)
            ]
        )
        return psi_total

    @staticmethod
    def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Kullback-Leibler Divergence.
        Assumes p and q are probability distributions (sum to 1).
        If raw data, use histograms first.
        """
        return stats.entropy(p, q)
