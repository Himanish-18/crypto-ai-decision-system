import pandas as pd
import numpy as np
from typing import List, Optional

class MetaLabeler:
    """
    v19 Meta-Labeler.
    Generates ground-truth labels for historical data to train the Meta-Model and Failure Detector.
    Definition of Success:
    - Signal direction matches Price direction at Horizon.
    - AND Return > Fees + Slippage.
    """
    def __init__(self, fee_pct: float = 0.0004, slippage_pct: float = 0.0001):
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        
    def compute_labels(self, candles: pd.DataFrame, signals: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compute labels for multiple horizons.
        If 'signals' is provided (Series of -1, 0, 1), labels are binary (1=Trade Won, 0=Trade Lost).
        If 'signals' is None, we generate theoretical labels (1=Long would win, -1=Short would win, 0=Neutral).
        """
        df = candles.copy()
        close = df["close"] if "close" in df.columns else df["btc_close"]
        
        # Horizons (assuming 1m or 5m data? Let's assume indices effectively)
        # If 5m data: 12 = 1h, 48 = 4h, 288 = 24h
        # If 1m data: 60 = 1h, 240 = 4h
        
        horizons = {
            "1h": 60,
            "4h": 240,
            "24h": 1440
        }
        
        labels_df = pd.DataFrame(index=df.index)
        
        for name, h in horizons.items():
            # Future Return
            ret_fwd = close.pct_change(h).shift(-h)
            
            # Threshold for profitability
            # Round trip = 2 * (fee + slip)
            cost = 2 * (self.fee_pct + self.slippage_pct)
            
            if signals is not None:
                # Meta-Labeling existing signals
                # Label 1 if: (Signal=1 and Ret > Cost) OR (Signal=-1 and Ret < -Cost)
                # Label 0 otherwise
                
                # Align signal
                sig = signals
                
                outcome = pd.Series(0, index=df.index)
                
                # Long Wins
                outcome.loc[(sig == 1) & (ret_fwd > cost)] = 1
                # Short Wins
                outcome.loc[(sig == -1) & (ret_fwd < -cost)] = 1
                
                labels_df[f"meta_label_{name}"] = outcome
                
            else:
                # Theoretical Labels (3-class)
                # 1 = Long Profitable
                # -1 = Short Profitable
                # 0 = Chop/Loss
                outcome = pd.Series(0, index=df.index)
                outcome.loc[ret_fwd > cost] = 1
                outcome.loc[ret_fwd < -cost] = -1
                
                labels_df[f"label_{name}"] = outcome
                
        return labels_df
