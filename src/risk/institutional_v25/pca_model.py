from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCARiskModel:
    """
    5-Factor PCA Risk Model.
    Decomposes asset returns into latent risk factors to identify systemic exposure.
    """

    def __init__(self, n_components: int = 5):
        self.pca = PCA(n_components=n_components)
        self.components = None
        self.explained_variance = None

    def fit(self, returns_matrix: pd.DataFrame):
        """
        Fit PCA on a matrix of asset returns (Time x Assets).
        """
        self.pca.fit(returns_matrix)
        self.components = self.pca.components_
        self.explained_variance = self.pca.explained_variance_ratio_

    def get_factor_exposures(self, asset_returns: pd.Series) -> np.ndarray:
        """
        Get exposure of a single asset to the 5 factors.
        """
        # Reshape for sklearn
        # Need to project asset returns onto components?
        # Typically: Factor Loadings = Cov(R_i, F_j)
        # Using transform for simplicity if trained on same universe
        pass

    def suggest_hedge(self, current_portfolio_exposure: np.ndarray) -> Dict[str, float]:
        """
        Suggest hedging action to neutralize first component (Systematic Market Risk).
        """
        market_factor_exposure = current_portfolio_exposure[0]

        # If High Positive Exposure to Factor 1 (Market)
        if market_factor_exposure > 0.5:
            return {"action": "SHORT_BTC_FUTURES", "ratio": market_factor_exposure}

        return {"action": "NONE", "ratio": 0.0}


class KellyCriterion:
    """
    Optimal Position Sizing.
    """

    @staticmethod
    def calculate_optimal_f(win_rate: float, reward_risk_ratio: float) -> float:
        """
        Full Kelly = p - q/b
        Adjusted (Half Kelly) recommended.
        """
        if reward_risk_ratio == 0:
            return 0.0
        return (win_rate * (reward_risk_ratio + 1) - 1) / reward_risk_ratio
