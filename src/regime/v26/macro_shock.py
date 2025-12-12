
import numpy as np

class MacroShockPredictor:
    """
    Detects macro shocks using Funding Rate and Open Interest anomalies.
    """
    
    @staticmethod
    def detect_leverage_unwind(funding_rates: np.ndarray, open_interest: np.ndarray) -> bool:
        """
        Signal: High Funding + Drop in OI -> Liquidation Cascade?
        Or: Negative Funding + High OI -> Short Squeeze risk?
        
        Returns True if shock likely.
        """
        if len(funding_rates) < 10: return False
        
        current_funding = funding_rates[-1]
        current_oi = open_interest[-1]
        
        funding_z = (current_funding - np.mean(funding_rates)) / (np.std(funding_rates) + 1e-9)
        oi_z = (current_oi - np.mean(open_interest)) / (np.std(open_interest) + 1e-9)
        
        # Extreme positive funding (> 3 sigma)
        if funding_z > 3.0:
            return True
        
        # Massive OI drop (> 3 sigma)
        if oi_z < -3.0:
            return True
            
        return False
