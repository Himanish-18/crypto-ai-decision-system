import pandas as pd
import numpy as np
from typing import Optional

class TrendDepth:
    """
    Calculates Trend Depth (TD) score [0, 1] to quantify trend strength and conviction.
    """
    def __init__(self):
        pass

    def calculate(self, df: pd.DataFrame) -> float:
        """
        Compute Trend Depth Score.
        
        Formula:
          TD = norm(ADX_14) * 0.4 
             + norm(EMA_20 > EMA_50 > EMA_200) * 0.3
             + norm(ATR_pct * Momentum_5) * 0.2
             + norm(price / vwap_20) * 0.1
             
        Requires: 'high', 'low', 'close', 'volume' columns.
        """
        if len(df) < 200:
            return 0.5 # Default neutral
            
        # 1. ADX (Average Directional Index)
        # Using simplified calc or assuming column exists?
        # Let's calculate manually to be robust or check for col.
        
        adx_val = 0.5
        if "adx" in df.columns:
            adx_val = df["adx"].iloc[-1] / 100.0 # Normalize 0-100 -> 0-1
        else:
            # Quick approx or 0.5
            # Implementing full ADX here might be heavy. 
            # Check main.py -> add_ta_indicators likely adds 'trend_adx'
            if "trend_adx" in df.columns:
                 adx_val = df["trend_adx"].iloc[-1] / 100.0
            else:
                 # Fallback: simple trend measure
                 adx_val = 0.5
        
        # 2. EMA Alignment (Trend Structure)
        # EMA 20 > 50 > 200 (Bull) or 20 < 50 < 200 (Bear)
        # We want "Trend Strength", so direction doesn't matter, just alignment.
        
        # Calculate EMAs if missing
        close = df["close"] if "close" in df.columns else df["btc_close"]
        
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema50 = close.ewm(span=50).mean().iloc[-1]
        ema200 = close.ewm(span=200).mean().iloc[-1]
        
        score_ema = 0.0
        # Bull Alignment
        if ema20 > ema50 > ema200:
            score_ema = 1.0
        # Bear Alignment
        elif ema20 < ema50 < ema200:
            score_ema = 1.0 # Strong Trend
        # Partial
        elif (ema20 > ema50) and (ema50 > ema200 * 0.999): # Near alignment
            score_ema = 0.5
        elif (ema20 < ema50) and (ema50 < ema200 * 1.001):
            score_ema = 0.5
            
        # 3. ATR * Momentum (Volatility Adjusted Momentum)
        # Relativize: Are we moving FAST compared to recent volatility?
        # Mom5 = (Price - Price_5)
        # Norm = Abs(Mom5) / (ATR * 5)
        # If we moved 5 ATRs in 5 bars -> Score 1.0
        
        mom5 = close.diff(5).iloc[-1]
        
        atr = 0.0
        if "atr" in df.columns: atr = df["atr"].iloc[-1]
        elif "volatility_atr" in df.columns: atr = df["volatility_atr"].iloc[-1]
        else:
            # Calc TR
            h = df["high"] if "high" in df.columns else df["btc_high"]
            l = df["low"] if "low" in df.columns else df["btc_low"]
            c_prev = close.shift(1)
            tr = np.maximum(h - l, np.abs(h - c_prev), np.abs(l - c_prev))
            atr = tr.rolling(14).mean().iloc[-1]
            
        if atr == 0: atr = 1.0
        
        norm_mom = min(abs(mom5) / (atr * 3.0), 1.0) # Cap at 1.0 (3 ATR move is strong)
        
        # 4. VWAP Deviation
        # Dist from VWAP -> Trend extension
        # If 'vwap' not present, use typically calculated vwap or SMA20 as proxy for mean
        vwap = ema20 # Fallback proxy if vwap missing for simplicity in this calculation context
        dist_vwap = abs(close.iloc[-1] - vwap) / vwap
        # 2% Deviation = 1.0 score
        norm_vwap = min(dist_vwap / 0.02, 1.0)
        
        # WEIGHTED SUM
        # TD = ADX*0.4 + EMA*0.3 + Mom*0.2 + VWAP*0.1
        td_score = (adx_val * 0.4) + (score_ema * 0.3) + (norm_mom * 0.2) + (norm_vwap * 0.1)
        
        return float(np.clip(td_score, 0.0, 1.0))
