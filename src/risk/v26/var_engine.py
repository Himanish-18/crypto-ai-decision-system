
import numpy as np
from typing import List, Optional

class VaREngine:
    """
    Institutional VaR Calculator.
    Supports:
    1. Historical VaR (HVaR)
    2. Liquidity Adjusted VaR (LVaR) 
    """
    
    @staticmethod
    def calculate_hvar(returns: np.ndarray, confidence_level: float = 0.99) -> float:
        """
        Historical VaR: Percentile of historical returns.
        """
        if len(returns) == 0: return 0.0
        return -np.percentile(returns, 100 * (1 - confidence_level))
        
    @staticmethod
    def calculate_lvar(hvar: float, current_price: float, bid_ask_spread: float, impact_cost: float) -> float:
        """
        Liquidity Adjusted VaR = VaR + Liquidation Cost
        Liquidation Cost = (Spread/2) + Market Impact
        """
        liquidation_cost = (bid_ask_spread / 2.0) + impact_cost
        lvar_value = hvar + (liquidation_cost / current_price)
        return lvar_value

    @staticmethod
    def parametric_var(returns: np.ndarray, confidence_level: float = 0.99) -> float:
        """
        Parametric (Normal) VaR = mu - z * sigma
        """
        if len(returns) < 2: return 0.0
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Z-score approx
        z = 2.33 if confidence_level == 0.99 else 1.65
        
        return -(mu - z * sigma)
