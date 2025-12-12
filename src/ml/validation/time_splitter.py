from typing import Generator, Tuple

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class PurgedTimeSeriesSplit:
    """
    Purged K-Fold Cross Validation for Finance (Lopez de Prado).
    Prevents leakage by purging samples between train and test sets.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 24):
        self.n_splits = n_splits
        self.purge_gap = purge_gap  # Gap in hours/indices

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // (self.n_splits + 1)

        # Expanding Window with Purge
        for i in range(1, self.n_splits + 1):
            train_end = i * fold_size
            test_start = train_end + self.purge_gap
            test_end = test_start + fold_size

            if test_end > n_samples:
                break

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx


class BlockingTimeSeriesSplit:
    """
    Splits data into K disjoint blocks (No expanding window).
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(self, X: np.ndarray, y=None, groups=None):
        n = len(X)
        k_fold_size = n // self.n_splits
        indices = np.arange(n)

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(start + 0.8 * k_fold_size)  # 80% Train, 20% Test within block

            yield indices[start:mid], indices[mid:stop]
