from typing import Dict, Tuple

import numpy as np
import pandas as pd


class DataQualityChecks:
    """
    Automated Data Quality Guards.
    """

    @staticmethod
    def check_missing_data(
        df: pd.DataFrame, threshold: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Fail if missing data exceeds threshold.
        """
        missing_pct = df.isnull().mean().max()
        if missing_pct > threshold:
            return False, f"Max Missing % {missing_pct:.2%} > threshold {threshold:.2%}"
        return True, "PASS"

    @staticmethod
    def check_schema(
        df: pd.DataFrame, expected_columns: Dict[str, str]
    ) -> Tuple[bool, str]:
        """
        Verify columns and types.
        """
        for col, dtype in expected_columns.items():
            if col not in df.columns:
                return False, f"Missing Column: {col}"
            # Type check placeholder
        return True, "PASS"

    @staticmethod
    def check_outliers(series: pd.Series, z_thresh: float = 5.0) -> Tuple[bool, str]:
        """
        Detect extreme Z-score outliers.
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        n_outliers = (z_scores > z_thresh).sum()
        if n_outliers > 0:
            return False, f"Found {n_outliers} outliers > {z_thresh} sigma"
        return True, "PASS"
