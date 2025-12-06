import numpy as np
import pandas as pd
from typing import Dict, Any

class PanicExitModel:
    """
    3-Head Shock Model for detecting Panic/Liquidity Events.
    Heads:
      A: Microstructure (Spread Spike Recognition)
      B: Orderflow (Volume Burst & Imbalance)
      C: Sentiment (Proxies)
    """
    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate if a Panic Exit is required.
        Returns:
            {
                "panic_score": float [0-1],
                "exit_signal": bool,
                "heads": {"micro": float, "flow": float, "sentiment": float},
                "reason": str
            }
        """
        if len(df) < 30:
            return {"panic_score": 0.0, "exit_signal": False, "heads": {}, "reason": "Insufficient Data"}
            
        latest = df.iloc[-1]
        
        # --- Head A: Microstructure (Spread Spike) ---
        # Logic: Sudden expansion in High-Low Range relative to ATR
        # Feature: Range / ATR
        
        # Calculate ATR if missing
        if "atr" in df.columns:
            atr = df["atr"].iloc[-1]
        else:
            # Simple ATR 14
            tr = np.maximum(
                df["high"] - df["low"], 
                np.abs(df["high"] - df["close"].shift(1)), 
                np.abs(df["low"] - df["close"].shift(1))
            )
            atr = tr.rolling(14).mean().iloc[-1]
            
        if atr == 0: atr = 1.0
        
        current_range = latest["high"] - latest["low"]
        spread_ratio = current_range / atr
        
        # Activation: Sigmoid-like logic. 
        # If range is > 3x ATR -> High Panic likelihood
        score_micro = np.clip((spread_ratio - 1.5) / 3.0, 0, 1)
        
        # --- Head B: Orderflow (Volume Burst & Imbalance) ---
        # Logic: Volume Spike (> 3x Avg) + Price Drop
        
        vol_avg = df["volume"].rolling(20).mean().iloc[-1]
        if vol_avg == 0: vol_avg = 1.0
        
        vol_ratio = latest["volume"] / vol_avg
        
        # Imbalance: Close near Low (Bearish) or High (Bullish Panic Buy?)
        # User defined "Panic Exit" usually implies Crash/Dump protection for Longs.
        # But could be Short Squeeze too.
        # Let's focus on Volatility/Instability.
        
        score_flow = np.clip((vol_ratio - 2.0) / 4.0, 0, 1)
        
        # Check Directionality for "Imbalance"
        # If huge volume but small body -> Indecision (Not necessarily panic exit, maybe reversal).
        # If huge volume + huge body -> Breakout/Crash.
        
        # --- Head C: Sentiment (Proxies) ---
        # Uses 'feat_fear_proxy' and 'feat_panic_proxy' calculated in SentimentFeatures
        # Or calculates on fly if missing.
        
        score_sent = 0.0
        if "feat_fear_proxy" in df.columns:
            # Fear Proxy is Z-Score. > 2.0 or < -2.0 is extreme.
            fear_z = abs(latest["feat_fear_proxy"])
            score_sent = np.clip((fear_z - 2.0) / 2.0, 0, 1)
            
        if "feat_panic_proxy" in df.columns:
             # Additive impact
             score_sent = max(score_sent, float(latest["feat_panic_proxy"]))
             
        # --- Fusion ---
        # Weighted Max Pooling?
        # A Panic Spike in ANY head is dangerous.
        # Let's use a soft-maxish approach or weighted average.
        
        # Weights: Microstructure (Fastest) > Orderflow > Sentiment (Lagging)
        weighted_score = (score_micro * 0.4) + (score_flow * 0.4) + (score_sent * 0.2)
        
        # Boost if multiple heads fire
        if (score_micro > 0.5) and (score_flow > 0.5):
            weighted_score = min(weighted_score * 1.5, 1.0)
            
        # Sensitivity Adjustment
        final_score = np.clip(weighted_score * self.sensitivity, 0, 1)
        
        # Decision
        exit_signal = final_score > 0.65
        
        heads = {
            "micro": float(score_micro),
            "flow": float(score_flow),
            "sentiment": float(score_sent)
        }
        
        reason = []
        if score_micro > 0.6: reason.append(f"Spread Spike ({spread_ratio:.1f}x ATR)")
        if score_flow > 0.6: reason.append(f"Vol Burst ({vol_ratio:.1f}x Avg)")
        if score_sent > 0.6: reason.append("Extreme Sentiment")
        
        return {
            "panic_score": float(final_score),
            "exit_signal": bool(exit_signal),
            "heads": heads,
            "reason": ", ".join(reason) if reason else "Normal"
        }
