from typing import Callable, Tuple

import numpy as np


class Bootstrap:
    """
    Statistical Bootstrapping Library.
    """

    @staticmethod
    def calculate_ci(
        data: np.ndarray, func: Callable, n_reps: int = 1000, alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Calculate Confidence Interval.
        """
        replicates = np.empty(n_reps)
        n = len(data)

        for i in range(n_reps):
            sample = np.random.choice(data, size=n, replace=True)
            replicates[i] = func(sample)

        lower = np.percentile(replicates, 100 * (alpha / 2))
        upper = np.percentile(replicates, 100 * (1 - alpha / 2))

        return lower, upper

    @staticmethod
    def stationary_bootstrap(data: np.ndarray, block_len: int = 10, n_reps: int = 1000):
        """
        Stationary Bootstrap for Time-Series (Politis & Romano).
        Recommended for financial data to preserve dependency.
        """
        # (Simplified Circular Block Bootstrap)
        n = len(data)
        replicates = []

        for _ in range(n_reps):
            # Construct sample from blocks
            indices = []
            while len(indices) < n:
                start = np.random.randint(0, n)
                indices.extend([(start + i) % n for i in range(block_len)])

            sample_idx = indices[:n]
            replicates.append(data[sample_idx])

        return np.array(replicates)
