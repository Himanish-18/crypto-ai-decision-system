import numpy as np
import pandas as pd
import logging
from sklearn.decomposition import PCA
from typing import Dict, Tuple

logger = logging.getLogger("pca_risk")

class PCAFactorModel:
    """
    v24 Institutional Factor Risk Model.
    Decomposes portfolio risk into Systematic (Principal Components) and Idiosyncratic risk.
    """
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.components_ = None
        self.explained_variance_ = None
        self.feature_names_ = None
        
    def fit(self, returns_df: pd.DataFrame):
        """
        Fit PCA model to returns.
        returns_df: Cleaned returns (pct_change).
        """
        if returns_df.empty:
            logger.warning("Empty returns passed to PCA.")
            return
            
        # Use available components limited by feature count
        n_features = returns_df.shape[1]
        n_comp = min(self.n_components, n_features)
        self.pca = PCA(n_components=n_comp)
        
        self.pca.fit(returns_df)
        self.feature_names_ = returns_df.columns
        self.components_ = self.pca.components_
        self.explained_variance_ = self.pca.explained_variance_ratio_
        
        logger.info(f"PCA Fit Complete. Top Factor Explains: {self.explained_variance_[0]*100:.2f}% Variance")
        
    def get_factor_loadings(self) -> pd.DataFrame:
        """
        Return correlations between assets and factors (Loadings).
        """
        if self.components_ is None:
            return pd.DataFrame()
            
        loadings = pd.DataFrame(
            self.components_.T * np.sqrt(self.pca.explained_variance_),
            columns=[f"Factor_{i+1}" for i in range(self.pca.n_components_)],
            index=self.feature_names_
        )
        return loadings
        
    def decompose_risk(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate Portfolio Risk contribution from Factors vs Idiosyncratic.
        weights: Dictionary of asset weights.
        """
        if self.components_ is None:
            return {"total_risk": 0.0}
            
        # Align weights to features
        w_vec = np.array([weights.get(asset, 0.0) for asset in self.feature_names_])
        
        # Factor Risk
        # Variance(Port) = w.T * (Beta * FactorCov * Beta.T + D) * w
        # PCA implies FactorCov is diagonal (eigenvalues)
        
        # Project weights onto factors
        factor_exposures = np.dot(w_vec, self.components_.T) # w * Beta^T ? No.
        # Component is (n_factors, n_features). w is (n_features,)
        # Exposure to factor k = Sum(w_i * Loading_ik)
        
        # Let's use simpler approx: Total Variance = Systematic + Idiosyncratic
        # Systematic Var = Sum( (Exposure_k)^2 * Var(Factor_k) )
        
        # Var(Factor_k) = self.pca.explained_variance_[k] (absolute variance, not ratio)
        var_factors = self.pca.explained_variance_
        
        # Projection of portfolio returns onto components
        # We need loadings.
        # Actually: Factor Covariance matrix in PCA basis is diag(explained_variance_)
        # Portfolio Factor Beta vector = w * components_.T 
        port_factor_betas = np.dot(w_vec, self.components_.T)
        
        systematic_var = np.sum(port_factor_betas**2 * var_factors)
        
        # Total Variance estimation (assuming data represents distribution)
        # Reconstruct Covariance
        # Cov ~ V . diag(var) . V.T
        # We don't have Disiosyncratic matrix easily without observing residuals.
        # We can approximate Total Portfolio Var if we had the Cov matrix.
        # But we only stored PCA.
        # Risk decomposition is usually: Total Risk - Systematic = Idiosyncratic.
        
        return {
            "systematic_variance": systematic_var,
            "top_factor_exposure": port_factor_betas[0]
        }
