import numpy as np
from typing import List, Dict

class AdvancedRiskModel:
    """
    v30 Institutional Risk: Component VaR & Marginal VaR.
    Decomposes portfolio risk into per-asset contributions.
    """
    
    @staticmethod
    def calculate_component_var(weights: np.ndarray, cov_matrix: np.ndarray, confidence=0.95):
        """
        Component VaR = w_i * marginal_VaR_i
        Marginal VaR = (Cov * w) / Portfolio_Vol
        """
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_var)
        
        # Z-score for 95%
        z = 1.65 
        
        marginal_var = np.dot(cov_matrix, weights) / port_vol * z
        component_var = weights * marginal_var
        
        return component_var, marginal_var, port_vol * z

    @staticmethod
    def stress_test_matrix(current_prices: np.ndarray, scenarios: List[float]):
        """
        Apply stress factors (e.g., [-0.20, -0.50, +0.10])
        """
        results = {}
        for shock in scenarios:
            shocked_prices = current_prices * (1 + shock)
            results[f"Shock {shock*100:.0f}%"] = shocked_prices
        return results
