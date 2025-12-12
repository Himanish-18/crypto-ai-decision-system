
import numpy as np
from typing import Dict, List

class AutoStressGrid:
    """
    Recalculates portfolio PnL under pre-defined stress scenarios.
    """
    
    SCENARIOS = {
        "FlashCrash_10pct": -0.10,
        "FlashCrash_20pct": -0.20,
        "CryptoWinter_50pct": -0.50,
        "VolSpike_200pct": 0.0, # Price neutral, Vol up
    }
    
    @staticmethod
    def run_stress_test(portfolio_nav: float, positions: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate NAV impact for each scenario.
        Assumes linear correlation (Beta=1) for simplicity in V1.
        """
        results = {}
        total_exposure = sum([abs(q * prices.get(s, 0)) for s, q in positions.items()])
        
        for name, price_shock in AutoStressGrid.SCENARIOS.items():
            # Assume 100% correlation for stress
            pnl_shock = total_exposure * price_shock
            
            # For Vol Spike (if we had options, this would be non-zero)
            if name.startswith("VolSpike") and price_shock == 0:
                 pnl_shock = 0.0 
                 
            simulated_nav = portfolio_nav + pnl_shock
            results[name] = simulated_nav
            
        return results
