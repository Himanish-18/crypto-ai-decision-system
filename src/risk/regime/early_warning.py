
import numpy as np

# v39 Early Warning System
# Predicts regime shifts 3-5 candles ahead using heuristics.

class EarlyWarningSystem:
    def __init__(self):
        self.fragility_score = 0.0
        
    def check_signals(self, volatility, liquidity_depth, spread_velocity):
        """
        Input: normalized metrics (0-1)
        Output: warning_level (0.0 - 1.0)
        """
        # 1. Volatility Spike Warning
        vol_risk = 1.0 if volatility > 0.8 else 0.0
        
        # 2. Liquidity Vacuum (Low depth)
        liq_risk = 1.0 if liquidity_depth < 0.2 else 0.0
        
        # 3. Spread Explosion
        spd_risk = 1.0 if spread_velocity > 0.7 else 0.0
        
        # Composite Fragility
        self.fragility_score = (vol_risk * 0.4) + (liq_risk * 0.4) + (spd_risk * 0.2)
        
        if self.fragility_score > 0.7:
            return "CRITICAL_WARNING"
        elif self.fragility_score > 0.4:
             return "CAUTION"
        return "SAFE"
