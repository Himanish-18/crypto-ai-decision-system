
import numpy as np
from typing import Tuple

class VolatilityTracker:
    """
    Tracks higher-order volatility stats.
    """
    
    @staticmethod
    def calculate_vol_of_vol(returns: np.ndarray, window: int = 24) -> float:
        """
        Volatility of Volatility.
        Returns Stdev of rolling Stdev.
        """
        if len(returns) < window * 2: return 0.0
        
        # Calculate rolling vol
        rolling_vols = []
        for i in range(len(returns) - window):
            chunk = returns[i:i+window]
            rolling_vols.append(np.std(chunk))
            
        return np.std(rolling_vols)

    @staticmethod
    def jump_risk_detector(returns: np.ndarray) -> float:
        """
        Bi-Power Variation (BV) vs Realized Variance (RV).
        Jump Component ~ max(RV - BV, 0)
        
        BV = (pi/2) * sum(|r_t| * |r_{t-1}|)
        """
        if len(returns) < 2: return 0.0
        
        abs_rets = np.abs(returns)
        rv = np.sum(returns**2)
        
        # Bi-power variation
        bv = (np.pi / 2.0) * np.sum(abs_rets[1:] * abs_rets[:-1])
        
        # Contribution of jumps to total variance
        jump_var = max(rv - bv, 0.0)
        
        return jump_var
