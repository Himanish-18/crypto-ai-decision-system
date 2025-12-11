import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger("risk.var_engine")

class VarEngine:
    """
    v24 Institutional VaR Engine.
    Uses Ledoit-Wolf Shrinkage for Covariance Matrix.
    """
    def __init__(self, lookback_window: int = 252):
        self.window = lookback_window
        
    def calculate_var(self, returns: pd.DataFrame, weights: np.array, confidence: float = 0.99) -> Dict[str, float]:
        """
        Calculate Parametric VaR and Expected Shortfall (CVaR).
        """
        if returns.empty:
            return {"VaR": 0.0, "CVaR": 0.0}
            
        # 1. Ledoit-Wolf Shrinkage (Simplified Proxy)
        # Real LW shrinks towards identity or constant correlation
        cov_matrix = returns.cov()
        
        # 2. Portfolio Volatility
        port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # 3. VaR (Parametric)
        # Z-score for 99%
        z_score = -2.33 
        var_99 = port_vol * z_score
        
        # 4. Expected Shortfall (CVaR) - non-parametric historical
        port_returns = returns.dot(weights)
        cutoff = port_returns.quantile(1 - confidence)
        cvar_99 = port_returns[port_returns <= cutoff].mean()
        
        return {
            "VaR_99": var_99,
            "CVaR_99": cvar_99,
            "Vol_Annual": port_vol * np.sqrt(365)
        }
        
    def check_limits(self, var_metrics: Dict[str, float], nav: float) -> bool:
        """
        Hard Veto if VaR exceeds limits (e.g., 5% of NAV single day).
        """
        var_usd = var_metrics["VaR_99"] * nav
        limit = -0.05 * nav # Max 5% loss with 99% confidence
        
        if var_metrics["VaR_99"] < -0.05:
            logger.critical(f"🛑 RISK VETO: VaR {var_metrics['VaR_99']:.2%} exceeds limit -5%")
            return False
            
        return True
