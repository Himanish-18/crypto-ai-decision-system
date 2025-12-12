from typing import Tuple

import numpy as np
from scipy import stats


class HypothesisTests:
    """
    Statistical Hypothesis Tests.
    """

    @staticmethod
    def t_test_one_sample(
        data: np.ndarray, popmean: float = 0.0
    ) -> Tuple[float, float]:
        """
        Test if mean is significantly different from popmean.
        """
        return stats.ttest_1samp(data, popmean)

    @staticmethod
    def rolling_stability_test(data: np.ndarray, window: int = 100) -> float:
        """
        Measures stability of mean via Coefficient of Variation of rolling means.
        Lower is more stable.
        """
        if len(data) < window:
            return 0.0
        series = np.array(data)
        means = []
        for i in range(len(data) - window):
            means.append(np.mean(series[i : i + window]))

        means = np.array(means)
        if np.mean(means) == 0:
            return 0.0
        return np.std(means) / abs(np.mean(means))
