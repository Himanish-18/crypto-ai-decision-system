import numpy as np
import pandas as pd
from scipy.stats import entropy


class EntropyFeatures:
    """
    v24 Entropy Features.
    Measures information content (uncertainty) in price/volume distributions.
    """

    @staticmethod
    def shannon_entropy(series: pd.Series, bins: int = 20) -> float:
        """Calculate Shannon Entropy of return distribution."""
        hist, _ = np.histogram(series, bins=bins, density=True)
        return entropy(hist + 1e-9)

    @staticmethod
    def compute_features(df: pd.DataFrame, windows=[20, 50, 100]) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        returns = df["close"].pct_change()

        for w in windows:
            features[f"entropy_{w}"] = returns.rolling(w).apply(
                lambda x: EntropyFeatures.shannon_entropy(x)
            )

        return features
