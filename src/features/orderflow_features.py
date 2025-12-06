import pandas as pd
import numpy as np

class OrderFlowFeatures:
    """
    Derives microstructure and order-flow proxies from OHLCV data.
    """

    def compute_all(self, df: pd.DataFrame, symbol: str = "btc") -> pd.DataFrame:
        """Compute all order flow features for a given symbol."""
        df = self._compute_imbalance_proxy(df, symbol)
        df = self._compute_wick_metrics(df, symbol)
        df = self._compute_volatility_proxies(df, symbol)
        df = self._compute_volume_shocks(df, symbol)
        return df

    def _compute_imbalance_proxy(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Estimate buy/sell pressure using Tick Rule Proxy.
        Logic: (Close - Open) / (High - Low) * Volume
        This gives a signed volume flow: Positive if Close > Open, scaled by range position.
        """
        o = df[f"{symbol}_open"]
        h = df[f"{symbol}_high"]
        l = df[f"{symbol}_low"]
        c = df[f"{symbol}_close"]
        v = df[f"{symbol}_volume"]

        range_ = h - l + 1e-9
        # Close location within range: (2* (C-L) - (H-L)) / (H-L) -> -1 to 1
        # If C=H, val=1. If C=L, val=-1.
        close_loc = (2 * (c - l) - range_) / range_
        
        df[f"{symbol}_of_buy_sell_imbalance"] = close_loc * v
        
        # Cumulative Flow (Flow 24h)
        df[f"{symbol}_of_flow_24h"] = df[f"{symbol}_of_buy_sell_imbalance"].rolling(24).sum()
        
        return df

    def _compute_wick_metrics(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        High-fidelity wick ratios to detect rejections/exhaustion.
        """
        o = df[f"{symbol}_open"]
        h = df[f"{symbol}_high"]
        l = df[f"{symbol}_low"]
        c = df[f"{symbol}_close"]
        
        body = (c - o).abs()
        upper_wick = h - np.maximum(c, o)
        lower_wick = np.minimum(c, o) - l
        total_range = h - l + 1e-9
        
        # Rejection Ratio: Max Wick / Total Range
        df[f"{symbol}_of_wick_reversal"] = np.maximum(upper_wick, lower_wick) / total_range
        
        # Wick Balance: (Upper - Lower) / Range
        # Positive = Selling Pressure (Upper wick large), Negative = Buying Pressure
        df[f"{symbol}_of_wick_balance"] = (upper_wick - lower_wick) / total_range
        
        return df

    def _compute_volatility_proxies(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Intrabar volatility estimation.
        """
        h = df[f"{symbol}_high"]
        l = df[f"{symbol}_low"]
        
        # Parkinson Volatility Proxy (High-Low scale)
        # Log(High/Low)^2 / (4 * log(2))
        const = 4 * np.log(2)
        log_hl = np.log(h / (l + 1e-9))
        df[f"{symbol}_of_intrabar_volatility"] = (log_hl ** 2) / const
        
        return df

    def _compute_volume_shocks(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Volume Z-Scores and expansion detection.
        """
        v = df[f"{symbol}_volume"]
        
        # Volume Z-Score (vs 20 period avg)
        v_mean = v.rolling(20).mean()
        v_std = v.rolling(20).std()
        df[f"{symbol}_of_volume_shock"] = (v - v_mean) / (v_std + 1e-9)
        
        # Volume Trend (Short vs Long)
        v_short = v.rolling(5).mean()
        v_long = v.rolling(50).mean()
        df[f"{symbol}_of_volume_trend"] = v_short / (v_long + 1e-9)
        
        return df
