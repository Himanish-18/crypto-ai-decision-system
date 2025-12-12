import logging
from collections import deque

import numpy as np
import pandas as pd

logger = logging.getLogger("deribit_vol")


class DeribitVolMonitor:
    """
    Monitors Deribit Options Market for Implied Volatility (IV) signals.
    Calculates 'OMX_VolFactor' (Aggregate Fear) and 'CrashPremium' (Skew).
    """

    def __init__(self, window_size=100):
        self.iv_history = deque(maxlen=window_size)
        self.skew_history = deque(maxlen=window_size)

    def fetch_iv_surface(self, spot_price: float):
        """
        Simulates fetching IV surface from Deribit.
        In production, this would call Deribit API.
        """
        # Simulated IV Source based on mock environment behavior
        # Base IV ~ 50% (0.50) + Random Noise

        base_iv = 0.50

        # Simulate Skew: Puts usually more expensive than Calls in crypto
        put_iv = base_iv + np.random.uniform(0.0, 0.1)
        call_iv = base_iv + np.random.uniform(-0.05, 0.05)
        atm_iv = base_iv

        return {"atm_iv": atm_iv, "otm_put_iv": put_iv, "otm_call_iv": call_iv}

    def get_metrics(self, spot_price: float) -> dict:
        """
        Returns {
            "omx_vol_factor": float, # 0.0 to 1.0 Normalized Vol Score
            "crash_premium": float,  # Put IV - Call IV
            "is_critical": bool      # If Vol > 95th Percentile
        }
        """
        surface = self.fetch_iv_surface(spot_price)
        atm_iv = surface["atm_iv"]
        skew = surface["otm_put_iv"] - surface["otm_call_iv"]

        self.iv_history.append(atm_iv)
        self.skew_history.append(skew)

        # Calculate Percentile
        is_critical = False
        vol_factor = 0.5

        if len(self.iv_history) > 10:
            iv_array = np.array(self.iv_history)
            p95 = np.percentile(iv_array, 95)
            p05 = np.percentile(iv_array, 5)

            if atm_iv > p95:
                is_critical = True

            # Normalize Vol Factor (0-1)
            if p95 > p05:
                vol_factor = (atm_iv - p05) / (p95 - p05)
            vol_factor = np.clip(vol_factor, 0.0, 1.0)

        return {
            "omx_vol_factor": float(vol_factor),
            "crash_premium": float(skew),
            "current_iv": float(atm_iv),
            "is_critical": is_critical,
        }
