
import numpy as np
from scipy.stats import skew, kurtosis, norm

class TailRiskEstimator:
    """
    Estimates extreme tail risk using advanced statistical methods.
    """
    
    @staticmethod
    def cornish_fisher_var(returns: np.ndarray, confidence: float = 0.99) -> float:
        """
        Adjusts Gaussian VaR for Skewness and Kurtosis.
        Z_cf = Z + (1/6)(Z^2 - 1)S + (1/24)(Z^3 - 3Z)K - (1/36)(2Z^3 - 5Z)S^2
        """
        if len(returns) < 10: return 0.0
        
        z = norm.ppf(1 - confidence) # e.g. -2.33
        s = skew(returns)
        k = kurtosis(returns) # Excess kurtosis
        
        z_cf = z + (1/6)*(z**2 - 1)*s + (1/24)*(z**3 - 3*z)*k - (1/36)*(2*z**3 - 5*z)*(s**2)
        
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        return -(mu + z_cf * sigma)

    @staticmethod
    def peaks_over_threshold(returns: np.ndarray, threshold_percentile: float = 5) -> float:
        """
        Simplified EVT approach: Average excess over a high threshold.
        (Expected Shortfall equivalent for the calculation)
        """
        if len(returns) == 0: return 0.0
        
        thresh = np.percentile(returns, threshold_percentile)
        tail_losses = returns[returns <= thresh]
        
        if len(tail_losses) == 0: return 0.0
        return np.mean(tail_losses)
