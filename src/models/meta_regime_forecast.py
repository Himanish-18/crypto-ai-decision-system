import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("meta_regime")

class MetaRegimeForecast:
    """
    Predicts Market Regime 3-6 steps ahead.
    Uses heuristic rules + simple ML (Simulated for v10).
    """
    def __init__(self):
        self.regimes = ["STABLE", "VOLATILE", "TRENDING", "CRASH_RISK"]
        
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Returns {
            "predicted_regime": str,
            "confidence": float
        }
        """
        if df.empty or len(df) < 20:
            return {"predicted_regime": "STABLE", "confidence": 0.0}
            
        latest = df.iloc[-1]
        
        # 1. Volatility Gradient (Are we heating up?)
        atr = latest.get("btc_atr_14", 0)
        atr_prev = df.iloc[-5].get("btc_atr_14", atr)
        
        vol_grad = (atr - atr_prev) / atr_prev if atr_prev > 0 else 0
        
        # 2. Entropy / Noise (e.g. from efficiency ratio if available)
        
        # Heuristic Logic (Placeholder for trained XGBClassifier)
        regime = "STABLE"
        conf = 0.5
        
        if vol_grad > 0.2:
            regime = "VOLATILE"
            conf = 0.7
            
        if vol_grad > 0.5:
             regime = "CRASH_RISK"
             conf = 0.8
             
        # Trend Strength
        tcn_prob = latest.get("tcn_prob", 0.5) # If available
        if abs(tcn_prob - 0.5) > 0.3:
            if regime != "CRASH_RISK":
                regime = "TRENDING"
                conf = 0.75
                
        return {
            "predicted_regime": regime,
            "confidence": conf,
            "vol_gradient": vol_grad
        }
