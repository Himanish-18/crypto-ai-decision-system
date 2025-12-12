
import logging
import numpy as np

# v39 Probabilistic Position Sizing
# Scales size based on Signal Confidence and Regime Volatility.

class ProbabilisticSizing:
    def __init__(self, base_size=1.0):
        self.base_size = base_size
        self.logger = logging.getLogger("risk_sizing")
        
    def get_size(self, signal_strength, volatility, prob_win):
        """
        Kelly Criterion-ish Logic adjusted for Regime.
        """
        # 1. Kelly Inputs
        # Win Rate ~ Prob Win
        # Reward/Risk Ratio (assumed 1.5 for now)
        b = 1.5
        p = prob_win
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0.0, kelly_fraction)
        
        # 2. Half-Kelly De-leveraging
        target_size = kelly_fraction * 0.5
        
        # 3. Volatility Adjustment (Inverse Vol)
        # If vol is high, reduce size
        vol_scalar = 1.0 / (1.0 + volatility * 10)
        
        final_size = target_size * vol_scalar * self.base_size
        
        self.logger.info(f"Sizing: Sig={signal_strength:.2f} Prob={prob_win:.2f} Vol={volatility:.2f} -> Size={final_size:.2f}")
        return final_size
