from typing import Any, Dict

import numpy as np
import pandas as pd


class SeasonalityEngine:
    """
    Extracts cyclic time features and historical volatility profiles
    to help models understand intraday seasonality (e.g., NY Open vs Asia Lunch).
    """

    def __init__(self):
        pass

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds cyclic time encodings and hourly vol stats.
        """
        if "timestamp" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            return df

        # Ensure DateTime
        if "timestamp" in df.columns:
            dt = pd.to_datetime(df["timestamp"])
        else:
            dt = df.index

        # 1. Cyclic Hour (Sine/Cosine)
        # 24 hour cycle
        df["feat_hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        df["feat_hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)

        # 2. Cyclic Day of Week
        # 7 day cycle
        df["feat_day_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df["feat_day_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # 3. Market Session Flag (One-Hot-ish or Categorical)
        # 0=Asia, 1=London, 2=NY, 3=Overlap
        # Crude approximation:
        # Asia: 00-08 UTC
        # London: 08-16 UTC
        # NY: 13-21 UTC

        # We use intersection logic
        # is_ny = ((dt.hour >= 13) & (dt.hour <= 21)).astype(int)
        # is_london = ((dt.hour >= 8) & (dt.hour <= 16)).astype(int)

        df["feat_is_ny_open"] = ((dt.hour >= 13) & (dt.hour <= 16)).astype(
            int
        )  # Overlap
        df["feat_is_asia_close"] = ((dt.hour >= 6) & (dt.hour <= 9)).astype(int)

        return df
