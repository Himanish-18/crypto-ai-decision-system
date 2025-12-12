import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger("regime_validation")


class RegimeValidator:
    """
    Validates model performance segmented by market regime.
    """

    @staticmethod
    def validate_by_regime(
        y_true: np.ndarray, y_pred: np.ndarray, regimes: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate accuracy/sharpe per regime.
        """
        df = pd.DataFrame({"true": y_true, "pred": y_pred, "regime": regimes})

        results = {}
        for regime_name, group in df.groupby("regime"):
            # Mock accuracy: correlation of true vs pred
            # Or simple direction match
            acc = np.mean(np.sign(group["true"]) == np.sign(group["pred"]))
            results[f"acc_{regime_name}"] = acc

        return results
