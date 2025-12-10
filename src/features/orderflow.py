import pandas as pd
import numpy as np
from typing import Optional

class OrderFlowFeatures:
    """
    Generates Order Flow and Market Intelligence features.
    Includes: CVD, Imbalance, Whale Sweeps, Dominance.
    """
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Pro v7 Order Flow features to DataFrame.
        """
        df = df.copy()

        # Define columns dynamically
        vol_col = "btc_volume" if "btc_volume" in df.columns else "volume"
        close_col = "btc_close" if "btc_close" in df.columns else "close"
        open_col = "btc_open" if "btc_open" in df.columns else "open"
        
        # 1. CVD (Cumulative Volume Delta)
        # Needs 'taker_buy_base_asset_volume' (Binance std) or buy_volume
        if "taker_buy_base_asset_volume" in df.columns:
            buy_vol = df["taker_buy_base_asset_volume"]
            sell_vol = df[vol_col] - buy_vol
            delta = buy_vol - sell_vol
            df["feat_cvd"] = delta.cumsum()
            df["feat_cvd_mom"] = df["feat_cvd"].diff(5)
            
            # 2. Imbalance (Bid/Ask) - Proxy using buy/sell pressure
            # Ratio of Buy vs Total
            df["feat_imbalance"] = (buy_vol - sell_vol) / (df[vol_col] + 1e-8)
            
        else:
            # Proxy: Weighted Price Direction
            # Close > Open -> Buy Vol dominated?
            direction = np.sign(df[close_col] - df[open_col])
            # 1. Delta (Aggressive Buying vs Selling)
            # Proxy: (Close - Open) * Volume * sign
            # Positive candle ~= Net Buying, Negative ~= Net Selling
            df["feat_of_delta"] = direction * df[vol_col]
            delta_proxy = df["feat_of_delta"]
            df["feat_cvd"] = delta_proxy.cumsum()
            df["feat_cvd_mom"] = df["feat_cvd"].diff(5)
            df["feat_imbalance"] = delta_proxy.rolling(5).sum() / (df[vol_col].rolling(5).sum() + 1e-8)

        # 3. Whale Sweep Detector
        # Aggressive Market Orders: High Vol / Low Count (if count avail) or High Vol spike relative to recent
        # Whale = Vol > 3 * Avg Vol AND High Wick/Body?
        
        vol_ma = df[vol_col].rolling(20).mean()
        df["feat_whale_idx"] = df[vol_col] / (vol_ma + 1e-8)
        # > 3.0 indicates "Sweep" or burst
        
        # 4. Spread Risk / Liquidity Toxicity
        # Proxy: Volatility / Volume (Illiquidity ratio) - Amihud
        ret = df[close_col].pct_change().abs()
        df["feat_liquidity_toxicity"] = ret / (df[vol_col] * df[close_col] + 1e-8) # Dollar vol
        
        # 5. BTC Dominance / Cross Asset
        # Assuming we only have BTC df here. If we had ETH, we'd join.
        # Fallback: Beta proxy?
        # We can predict "Dominance Regime" based on BTC strength relative to its own history?
        # Let's use simple RS (Relative Strength) proxy if ETH col missing.
        df["feat_btc_dom_change"] = 0.0 # Placeholder if no cross-data
        
        # Standardize features for NN
        cols_to_norm = ["feat_cvd_mom", "feat_imbalance", "feat_whale_idx", "feat_liquidity_toxicity"]
        for c in cols_to_norm:
            if c in df.columns:
                 # Rolling Z-score or MinMax
                 rm = df[c].rolling(100).mean()
                 rs = df[c].rolling(100).std()
                 df[c] = (df[c] - rm) / (rs + 1e-8)
        
        return df.fillna(0)
