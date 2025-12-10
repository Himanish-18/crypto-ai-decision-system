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
        
        if "fundingRate" not in df.columns or df["fundingRate"].isna().all():
            # Trigger Synthetic Fallback
            df = SentimentFeatures.calculate_synthetic_sentiment(df, window)
            # Use synthetic as fear proxy if needed or just return?
            # calculate_synthetic_sentiment sets 'feat_synthetic_sentiment'
            # and 'feat_sentiment_proxy' fallback.
            
            # We still might want 'feat_fear_proxy' filled with something?
            # Let's map synthetic to fear proxy inversely if needed or 0
            if "feat_fear_proxy" not in df.columns:
                 df["feat_fear_proxy"] = 0.0 # Neutral
            
            funding = pd.Series(0.0001, index=df.index)
        else:
            funding = df["fundingRate"]
            
        if "volume" in df.columns:
            vol_col = "volume"
        elif "btc_volume" in df.columns:
            vol_col = "btc_volume"
        else:
            vol_col = None
            
        # Vol * Abs(Funding)
        if vol_col:
             fear_base = df[vol_col] * funding.abs()
        else:
             fear_base = pd.Series(0, index=df.index)
        
        # Z-Score
        rolling_mean = fear_base.rolling(window=window).mean()
        rolling_std = fear_base.rolling(window=window).std()
        df["feat_fear_proxy"] = (fear_base - rolling_mean) / (rolling_std + 1e-8)
        
        # 2. Sentiment Proxy: Rolling Correlation(Price, Open Interest)
        # Logic: 
        #   Corr > 0: Price Up + OI Up (Bullish Confidence) or Price Down + OI Down (Long Liquidation)
        #   Corr < 0: Price Up + OI Down (Short Covering) or Price Down + OI Up (Aggressive Shorting)
        
        # Only compute if we didn't just run synthetic fallback (which populates feat_sentiment_proxy)
        if "feat_synthetic_sentiment" not in df.columns:
            if "openInterest" not in df.columns:
                # Fallback based on volume accumulation as proxy for interest
                if vol_col:
                    oi_proxy = df[vol_col].cumsum()
                else:
                    oi_proxy = pd.Series(0, index=df.index)
                
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
        elif "volume" not in df.columns: vol_col = None
        
        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns or vol_col is None:
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

    @staticmethod
    def calculate_synthetic_sentiment(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
        """
        Generate a synthetic sentiment score (-1 to 1) using only OHLCV data.
        Used as fallback when Funding Rate or IV is missing.
        Components:
        1. Rolling Skewness (Market Bias)
        2. Volatility Shock (Abs(Ret) / ATR)
        3. Wick Ratios (Liquidation/Selling Pressure)
        """
        df = df.copy()
        
        # Helper: Column mapping
        close_col = "close" if "close" in df.columns else "btc_close"
        high_col = "high" if "high" in df.columns else "btc_high"
        low_col = "low" if "low" in df.columns else "btc_low"
        open_col = "open" if "open" in df.columns else "btc_open"
        
        # 1. Rolling Skewness (Bias)
        # Positive Skew -> More positive outliers -> Bullish bias? 
        # Actually in finance, negative skew often implies crash risk (fear).
        # Positive skew implies pump potential.
        skew = df[close_col].rolling(window).skew().fillna(0)
        # Normalize Skew (-1 to 1 mostly)
        skew_score = np.tanh(skew)
        
        # 2. Volatility Shock (Momentum Strength)
        # Abs(Ret) / ATR
        # If Ret > 0: Bullish Shock. If Ret < 0: Bearish Shock.
        tr1 = df[high_col] - df[low_col]
        tr2 = (df[high_col] - df[close_col].shift(1)).abs()
        tr3 = (df[low_col] - df[close_col].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().replace(0, 1e-8)
        
        ret = df[close_col].diff()
        vol_shock = ret / atr
        vol_shock_score = np.tanh(vol_shock) # -1 to 1
        
        # 3. Wick Ratios (Pressure)
        # Upper Wick: (High - Max(Open, Close)) / Range
        # Lower Wick: (Min(Open, Close) - Low) / Range
        # Range = High - Low
        range_ = (df[high_col] - df[low_col]).replace(0, 1e-8)
        body_top = pd.concat([df[open_col], df[close_col]], axis=1).max(axis=1)
        body_bottom = pd.concat([df[open_col], df[close_col]], axis=1).min(axis=1)
        
        upper_wick = (df[high_col] - body_top) / range_
        lower_wick = (body_bottom - df[low_col]) / range_
        
        # Net Wick Pressure: Lower (Bullish) - Upper (Bearish)
        wick_score = lower_wick - upper_wick # -1 (Bearish) to 1 (Bullish)
        
        # Composite Score
        # Weights: Skew (0.3), Vol Shock (0.4), Wick (0.3)
        composite = (skew_score * 0.3) + (vol_shock_score * 0.4) + (wick_score * 0.3)
        
        # Store in the standard 'feat_sentiment_proxy' column used by engine
        # But we name it distinct to indicate synthetic origin if needed. 
        # For compatibility with engine that looks for 'feat_sentiment_proxy':
        df["feat_synthetic_sentiment"] = composite
        
        # If the main proxy is missing, overwrite it
        if "feat_sentiment_proxy" not in df.columns or df["feat_sentiment_proxy"].isna().all():
             df["feat_sentiment_proxy"] = composite
             
        return df
