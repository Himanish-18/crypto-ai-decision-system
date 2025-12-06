import pandas as pd
import numpy as np

class SentimentFeatures:
    """
    Generates sentiment proxies from market data when live sentiment APIs are unavailable.
    """
    
    @staticmethod
    def calculate_proxies(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Add sentiment proxy columns to the DataFrame.
        Requires: 'close', 'volume', 'fundingRate' (optional), 'openInterest' (optional).
        """
        df = df.copy()
        
        # 1. Fear Proxy: Z-Score of (Volume * Abs(Funding))
        # Logic: High Volume + High Funding payments = Extreme Positioning/Fear/Greed
        
        # Funding Rate fallback
        if "fundingRate" not in df.columns:
            # Create dummy if missing (neutral)
            funding = pd.Series(0.0001, index=df.index)
        else:
            funding = df["fundingRate"]
            
        # Vol * Abs(Funding)
        fear_base = df["volume"] * funding.abs()
        
        # Z-Score
        rolling_mean = fear_base.rolling(window=window).mean()
        rolling_std = fear_base.rolling(window=window).std()
        df["feat_fear_proxy"] = (fear_base - rolling_mean) / (rolling_std + 1e-8)
        
        # 2. Sentiment Proxy: Rolling Correlation(Price, Open Interest)
        # Logic: 
        #   Corr > 0: Price Up + OI Up (Bullish Confidence) or Price Down + OI Down (Long Liquidation)
        #   Corr < 0: Price Up + OI Down (Short Covering) or Price Down + OI Up (Aggressive Shorting)
        
        if "openInterest" not in df.columns:
            # Fallback based on volume accumulation as proxy for interest
            oi_proxy = df["volume"].cumsum()
            
            if "btc_close" in df.columns:
                close_col = "btc_close"
            else:
                close_col = "close"
                
            df["feat_sentiment_proxy"] = df[close_col].rolling(window).corr(oi_proxy)
        else:
            if "btc_close" in df.columns:
                 close_col = "btc_close" 
            else:
                 close_col = "close"
            df["feat_sentiment_proxy"] = df[close_col].rolling(window).corr(df["openInterest"])
            
        # 3. Panic Proxy: Spread * Liquidations (Volume Imbalance)
        # Logic: Wide Spread + High Volume = Panic/Liquidity Gap
        
        # Determine column names
        high_col = "high"
        low_col = "low"
        close_col = "close"
        open_col = "open"
        vol_col = "volume"
        
        if "btc_high" in df.columns: high_col = "btc_high"
        if "btc_low" in df.columns: low_col = "btc_low"
        if "btc_close" in df.columns: close_col = "btc_close"
        if "btc_open" in df.columns: open_col = "btc_open"
        if "btc_volume" in df.columns: vol_col = "btc_volume"
        
        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
             # Cannot calculate spread, set neutral
             df["feat_panic_proxy"] = 0.0
             return df.fillna(0)

        # Spread proxy: High - Low (Range) if bid/ask spread not avail
        spread_proxy = (df[high_col] - df[low_col]) / df[close_col]
        
        # Liquidation proxy: Volume * (Close - Open) / Open (Price Impact)
        # High impact per unit volume -> Low Liquidity / Panic
        to_turn = df[vol_col] * (df[close_col] - df[open_col]).abs() / df[open_col]
        
        df["feat_panic_proxy"] = spread_proxy * to_turn
        
        # Normalize Panic Proxy (0-1 approx range via tanh)
        df["feat_panic_proxy"] = np.tanh(df["feat_panic_proxy"])
        
        return df.fillna(0)
