import numpy as np
import pandas as pd
import ta


class AlphaSignals:
    """
    Advanced Alpha Feature Set extraction module.
    Includes Formulaic Alphas, Microstructure Proxies, and Cross-Asset Signals.
    Total: ~20-30 Alphas.
    """

    def compute_all(self, df: pd.DataFrame, symbol: str = "btc") -> pd.DataFrame:
        """Compute all alpha signals for a given symbol."""
        df = self._compute_formulaic(df, symbol)
        df = self._compute_microstructure(df, symbol)
        df = self._compute_technical_alphas(df, symbol)
        df = self._compute_v5_alphas(df, symbol)

        # Cross-asset requires both BTC and ETH in df
        if "btc_close" in df.columns and "eth_close" in df.columns:
            df = self._compute_cross_asset(df)

        return df

    def _ts_rank(self, series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

    def _compute_formulaic(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Kakushadze 101 subset (Price/Volume)."""
        o = df[f"{symbol}_open"]
        h = df[f"{symbol}_high"]
        l = df[f"{symbol}_low"]
        c = df[f"{symbol}_close"]
        v = df[f"{symbol}_volume"]
        vwap = (h + l + c) / 3  # Simple proxy

        # Alpha#6: -1 * corr(open, volume, 10)
        df[f"{symbol}_alpha_006"] = -1 * o.rolling(10).corr(v)

        # Alpha#9: Close delta momentum
        delta_c = c.diff(1)
        cond1 = delta_c.rolling(5).min() > 0
        cond2 = delta_c.rolling(5).max() < 0
        df[f"{symbol}_alpha_009"] = np.where(
            cond1, delta_c, np.where(cond2, delta_c, -delta_c)
        )

        # Alpha#12: Sign(Vol Delta) * -1 * Close Delta
        df[f"{symbol}_alpha_012"] = np.sign(v.diff(1)) * (-1 * c.diff(1))

        # Alpha#26: High-Vol correlation
        # -1 * ts_max(corr(ts_rank(v, 5), ts_rank(h, 5), 5), 3)
        rank_v = self._ts_rank(v, 5)
        rank_h = self._ts_rank(h, 5)
        df[f"{symbol}_alpha_026"] = -1 * rank_v.rolling(5).corr(rank_h).rolling(3).max()

        # Alpha#41: VWAP Reversal (sqrt(h*l) - vwap)
        df[f"{symbol}_alpha_041"] = np.sqrt(h * l) - vwap

        # Alpha#53: Inner candle strength
        inner = ((c - l) - (h - c)) / (h - l + 1e-9)
        df[f"{symbol}_alpha_053"] = -1 * inner.diff(9)

        # Alpha#101: Candle Body / Range
        df[f"{symbol}_alpha_101"] = (c - o) / (h - l + 1e-9)

        # --- NEW FEATURES (Regime Upgrade) ---
        # 1. Price/Volume Correlations
        # Corr(Open, Volume) - 10 day rolling
        df[f"{symbol}_alpha_corr_open_vol"] = o.rolling(10).corr(v)
        # Corr(High, Volume) - 10 day rolling
        df[f"{symbol}_alpha_corr_high_vol"] = h.rolling(10).corr(v)

        # 2. Intrabar Momentum: (Close - Open) / Range
        df[f"{symbol}_alpha_intrabar_mom"] = (c - o) / (h - l + 1e-9)

        return df

    def _compute_microstructure(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Market Microstructure Proxies."""
        o = df[f"{symbol}_open"]
        h = df[f"{symbol}_high"]
        l = df[f"{symbol}_low"]
        c = df[f"{symbol}_close"]
        v = df[f"{symbol}_volume"]

        # 1. Trade Imbalance Proxy: (2*(Close-Low)-(High-Low))/(High-Low) * Volume
        # Captures buying/selling pressure within the candle
        range_ = h - l + 1e-9
        ad = (((c - l) - (h - c)) / range_) * v
        df[f"{symbol}_alpha_trade_imbalance"] = ad

        # 2. VWAP Reversal Alpha: (Price - VWAP) / ATR
        # Measures deviation from value. High pos -> Overbought (Revert?), High neg -> Oversold
        vwap = (h + l + c) / 3
        atr = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        df[f"{symbol}_alpha_vwap_reversal"] = (c - vwap) / (atr + 1e-9)

        # 3. Volume Imbalance Alpha:
        # Up-Vol vs Down-Vol ratio over window
        ret = c.pct_change()
        up_vol = v.where(ret > 0, 0).rolling(24).sum()
        down_vol = v.where(ret < 0, 0).rolling(24).sum()
        df[f"{symbol}_alpha_vol_imbalance"] = (up_vol - down_vol) / (
            up_vol + down_vol + 1e-9
        )

        # 4. VPIN Proxy (Volume-Synchronized Probability of Informed Trading)
        # Simplified: Std(Volume * Return) / Mean(Volume)
        # Represents "Toxic Flow" or high conviction moves
        # We'll use: Rolling Std(Abs(Ret)*Vol)
        df[f"{symbol}_alpha_vpin_proxy"] = (ret.abs() * v).rolling(24).std()

        return df

    def _compute_technical_alphas(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Technical Analysis based Alphas."""
        c = df[f"{symbol}_close"]
        h = df[f"{symbol}_high"]
        l = df[f"{symbol}_low"]
        o = df[f"{symbol}_open"]
        v = df[f"{symbol}_volume"]

        # --- SIDEWAYS ALPHAS (Mean Reversion / Range) ---

        # 1. RSI Divergence Strength (Normalized)
        # RSI - EMA(RSI). High > Overbought vs Trend.
        rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
        rsi_ema = rsi.ewm(span=9).mean()
        df[f"{symbol}_alpha_rsi_divergence"] = (rsi - rsi_ema) / 100.0

        # 2. Bollinger Band Squeeze (Low Volatility -> Breakout?)
        # Or Width (Range check). For Sideways, we want to fade the bands.
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        # Fix: 'bollinger_mband' -> 'bollinger_mavg' (common in `ta`)
        width = (bb.bollinger_hband() - bb.bollinger_lband()) / (
            bb.bollinger_mavg() + 1e-9
        )
        # Rolling min of width to detect squeeze?
        # Let's use Width Z-Score. Low Z-Score = Squeeze.
        width_z = (width - width.rolling(20).mean()) / (width.rolling(20).std() + 1e-9)
        df[f"{symbol}_alpha_bb_width_squeeze"] = width_z

        # 3. Stochastic RSI (Oscillator)
        # Excellent for range bound.
        stoch_rsi = ta.momentum.StochRSIIndicator(c, window=14).stochrsi()
        # Centered around 0.5
        df[f"{symbol}_alpha_stoch_rsi"] = stoch_rsi - 0.5

        # --- HIGH VOLATILITY ALPHAS (Structure / Panic) ---

        # 4. Volatility Clustering / Ratio
        # ATR(5) / ATR(20). > 1 means Vol is expanding rapidly.
        atr5 = ta.volatility.AverageTrueRange(h, l, c, window=5).average_true_range()
        atr20 = ta.volatility.AverageTrueRange(h, l, c, window=20).average_true_range()
        df[f"{symbol}_alpha_atr_ratio"] = atr5 / (atr20 + 1e-9)

        # 5. Wick Ratios (Reversal signals in high vol)
        # Max(Upper, Lower) / Body
        body_size = (c - o).abs()
        upper_wick = h - np.maximum(c, o)
        lower_wick = np.minimum(c, o) - l
        # Use simple ratio
        max_wick = np.maximum(upper_wick, lower_wick)
        df[f"{symbol}_alpha_wick_ratio"] = max_wick / (body_size + 1e-9)

        # 6. Panic Candle (Volume * Price Move)
        # Normalized by recent average
        price_move = (c - o).abs()
        dollar_vol = price_move * v
        avg_dollar_vol = dollar_vol.rolling(20).mean()
        # Ratio > 3 implies Panic/Climax
        df[f"{symbol}_alpha_panic_candle"] = dollar_vol / (avg_dollar_vol + 1e-9)

        # OLDER Alphas (Kept for compatibility)
        # ADX Trend Alpha
        adx = ta.trend.ADXIndicator(h, l, c, window=14)
        adx_val = adx.adx()
        plus_di = adx.adx_pos()
        minus_di = adx.adx_neg()
        trend_dir = np.where(plus_di > minus_di, 1, -1)
        df[f"{symbol}_alpha_adx_trend"] = adx_val * trend_dir

        return df

    def _compute_v5_alphas(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        compute v5 alphas (Order Flow, Smart Money, Microstructure).
        """
        o = df[f"{symbol}_open"]
        h = df[f"{symbol}_high"]
        l = df[f"{symbol}_low"]
        c = df[f"{symbol}_close"]
        v = df[f"{symbol}_volume"]

        # 1. Order Flow Imbalance (Volume-weighted direction)
        # (BuyVol - SellVol) / TotalVol. Approx: Sign(Close-Open) * Volume
        # Simple proxy:
        direction = np.sign(c - o)
        of_imbalance = (direction * v).rolling(window=10).sum() / (
            v.rolling(window=10).sum() + 1e-9
        )
        df[f"{symbol}_alpha_of_imbalance"] = of_imbalance

        # 2. VWAP Deviation Z-Score
        # Reset VWAP daily? For continuous, we use rolling VWAP proxy or just mean price weighted
        # Let's use Rolling VWAP (24h)
        rolling_vwap = (v * (h + l + c) / 3).rolling(24).sum() / (
            v.rolling(24).sum() + 1e-9
        )
        std_24 = c.rolling(24).std()
        df[f"{symbol}_alpha_vwap_zscore"] = (c - rolling_vwap) / (std_24 + 1e-9)

        # 3. Smart Money Delta (Big Trades vs Small Trades)
        # Proxy: Volume^2 (emphasizes big candles) * Direction
        # "Smart Money" often moves in high volume, low volatility candles (accumulation)
        # Or high volume spikes. Let's use: (Vol / RollingMeanVol)^2 * Sign(Ret)
        rel_vol = v / (v.rolling(20).mean() + 1e-9)
        df[f"{symbol}_alpha_smart_money_delta"] = (rel_vol**2) * np.sign(c.diff())

        # 4. Micro-Trend Curl (Derivative of Moving Average Slope)
        # MA(10) slope change
        ma10 = c.rolling(10).mean()
        slope = ma10.diff()
        curl = slope.diff()  # Acceleration
        df[f"{symbol}_alpha_micro_curl"] = curl

        # 5. Volume-Flow RSI
        # RSI calculated on (Signed Volume)
        # Better: UpVol vs DownVol RSI
        up_vol = v.where(c > o, 0)
        down_vol = v.where(c <= o, 0)
        # Standard RSI formula on Up/Down averages
        rs = up_vol.rolling(14).mean() / (down_vol.rolling(14).mean() + 1e-9)
        df[f"{symbol}_alpha_vol_flow_rsi"] = 100 - (100 / (1 + rs))

        # 6. Liquidity Shock Index (Volatility * Spread proxy)
        # Amihud Illiquidity: |Ret| / (Price * Vol).
        # Shock: Rapid increase in illiquidity
        illiquidity = c.pct_change().abs() / (v * c + 1e-9)
        df[f"{symbol}_alpha_liquidity_shock"] = illiquidity.diff(3)

        # 7. Regime-Adaptive MACD
        # If Volatility High, Fast=5, Slow=15. If Low, Fast=12, Slow=26.
        # We compute both and blend based on ATR Ratio?
        # Or just compute "Fast MACD" as a separate alpha.
        ema5 = c.ewm(span=5).mean()
        ema15 = c.ewm(span=15).mean()
        df[f"{symbol}_alpha_macd_fast"] = ema5 - ema15

        # 8. Fair Value Gap (FVG) Strength
        # Bullish FVG: Low[t-2] > High[t]
        # Bearish FVG: High[t-2] < Low[t]
        # Magnitude = Gap Size / ATR
        # Let's stick to simple "Gap" feature: (Open - Close_prev) / ATR
        df[f"{symbol}_alpha_gap_strength"] = (o - c.shift(1)) / (
            c.rolling(24).std() + 1e-9
        )

        # 9. Momentum Decay
        # ROC(10) vs MACD Histogram. Divergence?
        # Histogram slope vs Price slope
        roc = c.pct_change(10)
        # Reuse standard MACD
        macd = ta.trend.MACD(c)
        hist = macd.macd_diff()
        # If Price Rising (ROC > 0) but Hist Falling (Diff < 0) -> Decay
        decay = np.where(
            (roc > 0) & (hist.diff() < 0),
            1,
            np.where((roc < 0) & (hist.diff() > 0), -1, 0),
        )
        df[f"{symbol}_alpha_momentum_decay"] = decay

        return df

    def _compute_cross_asset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-Asset Alphas (BTC vs ETH)."""
        # 1. Return Spread Z-Score
        diff = df["btc_ret"] - df["eth_ret"]
        mean_diff = diff.rolling(24).mean()
        std_diff = diff.rolling(24).std()
        df["alpha_cross_asset_spread"] = (diff - mean_diff) / (std_diff + 1e-9)

        # 2. Lead-Lag Correlation
        # Corr(BTC_t, ETH_t-1) vs Corr(BTC_t-1, ETH_t)
        # If BTC leads, Corr(ETH_t, BTC_t-1) should be high
        # We compute rolling correlation of lagged returns
        # Lag 1
        corr_btc_lead = df["btc_ret"].shift(1).rolling(24).corr(df["eth_ret"])
        corr_eth_lead = df["eth_ret"].shift(1).rolling(24).corr(df["btc_ret"])

        df["alpha_lead_lag"] = corr_btc_lead - corr_eth_lead  # Positive if BTC leads

        # New: Rolling Lead-Lag Variations
        # Difference in returns
        df["alpha_btc_eth_ret_diff"] = df["btc_ret"] - df["eth_ret"]

        # Spread Mean Reversion (Z-Score of spread)
        # Already done above as alpha_cross_asset_spread, but clarifying:
        # Spread = BTC Ret - ETH Ret. Z-Score = (Spread - Mean) / Std.
        # This is a solid mean-reversion signal.

        # ETH Ret - Beta * BTC Ret
        # Compute rolling beta
        cov = df["eth_ret"].rolling(24).cov(df["btc_ret"])
        var = df["btc_ret"].rolling(24).var()
        beta = cov / (var + 1e-9)
        df["alpha_eth_idiosyncratic"] = df["eth_ret"] - (beta * df["btc_ret"])

        return df
