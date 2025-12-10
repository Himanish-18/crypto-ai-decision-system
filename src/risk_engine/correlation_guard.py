
import pandas as pd
import numpy as np
import logging
from collections import deque

logger = logging.getLogger("correlation_guard")

class CorrelationGuard:
    """
    Monitors rolling correlation between assets (BTC, ETH, SOL, LTC).
    Reduces position sizing and suggests hedging if systematic risk (correlation) is high.
    Window: 30 minutes (Approx 6 samples of 5m candles, or more if 1m).
    Assumes 5m candles -> 6 samples is very short for correlation. 
    Let's use a larger window of observed data points irrespective of candle time if possible,
    or assume we get 1m updates? 
    If Live Engine runs on 5m candles, 30m = 6 points. That's statistically weak.
    Let's store last 30 data points (2.5 hours of 5m data) for robust correlation.
    """
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.assets = ["BTC", "ETH", "SOL", "LTC"]
        # Buffers: Dict of Deques
        self.buffers = {asset: deque(maxlen=window_size) for asset in self.assets}
        self.last_matrix = None
        
    def update(self, price_dict):
        """
        Update buffers with latest search.
        price_dict: {"BTC": 90000, "ETH": 3000, ...}
        """
        # Ensure we have data for all assets to keep alignment
        # If an asset is missing, we might skip update or append None (handled later)?
        # For simplicity, we assume synchronized updates.
        
        valid = True
        for asset in self.assets:
            if asset not in price_dict:
                valid = False
                break
        
        if valid:
            for asset in self.assets:
                self.buffers[asset].append(price_dict[asset])
        else:
            # Partial update? Better not to update to avoid misalignment
            pass

    def compute_matrix(self):
        """
        Compute Pearson correlation matrix of returns.
        """
        if len(self.buffers["BTC"]) < 10: # Need minimum samples
            return None
            
        data = {asset: list(self.buffers[asset]) for asset in self.assets}
        df = pd.DataFrame(data)
        
        # We need correlation of RETURNS, not prices (non-stationary)
        returns = df.pct_change().dropna()
        
        if len(returns) < 5:
            return None
            
        corr_matrix = returns.corr(method='pearson')
        self.last_matrix = corr_matrix
        return corr_matrix

    def calculate_risk_modifiers(self, current_pos_size=1.0):
        """
        Returns:
            size_scalar (float): Multiplier for position size (0.5 to 1.0).
            hedge_signal (bool): Whether to engage hedging.
            debug_info (dict): Details.
        """
        matrix = self.compute_matrix()
        if matrix is None:
            return 1.0, False, {"status": "insufficient_data"}
            
        # exclude self-correlation (diagonal)
        # Get off-diagonal elements
        mask = np.ones(matrix.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        off_diag = matrix.values[mask]
        
        max_corr = off_diag.max() if len(off_diag) > 0 else 0
        avg_corr = off_diag.mean() if len(off_diag) > 0 else 0
        
        size_scalar = 1.0
        hedge_signal = False
        
        # 1. Size Reduction Logic
        # If Max Correlation > 0.6, we reduce size.
        # Linearly scale from 1.0 at 0.6 to 0.5 at 0.9+
        if max_corr > 0.6:
            # excessive correlation
            # reduction = (max_corr - 0.6) / (0.9 - 0.6) * 0.5
            # Simplified:
            reduction_factor = max(0.0, (max_corr - 0.6) * 1.66) # 0.3 diff -> 0.5 reduction
            size_scalar = max(0.5, 1.0 - reduction_factor)
            logger.info(f"🔗 High Correlation ({max_corr:.2f}). Reducing Size Scalar to {size_scalar:.2f}")

        # 2. Hedging Logic
        # If Avg Correlation > 0.7 AND Position Size is significant (> 0.5x leverage/allocation)
        if avg_corr > 0.7 and current_pos_size > 0.5:
            hedge_signal = True
            logger.warning(f"🔗 Systemic Risk (Avg Corr {avg_corr:.2f}). HEDGING SIGNAL ACTIVE.")
            
        return size_scalar, hedge_signal, {
            "max_corr": max_corr,
            "avg_corr": avg_corr,
            "size_scalar": size_scalar,
            "hedge_signal": hedge_signal
        }
