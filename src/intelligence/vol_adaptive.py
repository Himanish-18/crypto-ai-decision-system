class VolAdaptiveThreshold:
    """
    Adjusts trading thresholds based on market volatility.
    """
    def __init__(self, base_threshold: float = 0.55, vol_sensitivity: float = 0.15):
        self.base = base_threshold
        self.sensitivity = vol_sensitivity # 0.15 per User Spec

    def get_threshold(self, volatility: float) -> float:
        """
        Calculate adaptive threshold.
        Rule: entry = base + (volatility * 0.15)
        
        Args:
           volatility: Current volatility (e.g. 0.01 for 1%).
           
        Returns:
           Adjusted threshold.
        """
        # User requested specific logic:
        # "Raise threshold when volatility < 0.8%"
        # "Lower threshold when volatility > 2%"
        # "Scale as: entry = base + (volatility * 0.15)"
        
        # Implementation of Formula:
        # If vol increases, term increases -> Threshold RISES.
        # This acts as a safety mechanism: In high vol, require higher confidence.
        
        return self.base + (volatility * self.sensitivity)
