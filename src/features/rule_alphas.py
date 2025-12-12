import numpy as np
import pandas as pd
import ta


class RuleAlphas:
    """
    Implementation of Rule-Based Alphas from Bits PDFs (OBV, Heikin-Ashi, Correlation).
    """

    def compute_all(self, df: pd.DataFrame, symbol: str = "btc") -> pd.DataFrame:
        df = self._calc_obv_trend(df, symbol)
        df = self._calc_heikin_atr(df, symbol)

        # Correlation requires both btc and eth cols to be present
        if "btc_close" in df.columns and "eth_close" in df.columns:
            df = self._calc_btc_eth_corr(df)

        return df

    def _calc_obv_trend(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """OBV Trend Following Strategy: OBV > EMA(OBV) -> Bullish."""
        # On Balance Volume
        obv = ta.volume.OnBalanceVolumeIndicator(
            close=df[f"{symbol}_close"], volume=df[f"{symbol}_volume"]
        ).on_balance_volume()

        # EMA of OBV
        obv_ema = obv.ewm(span=20).mean()

        # Alpha: 1 if OBV > EMA else -1
        # Smoothed signal
        df[f"{symbol}_alpha_obv_trend"] = np.where(obv > obv_ema, 1, -1)
        return df

    def _calc_heikin_atr(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Heikin-Ashi + ATR Trailing Stop Logic."""
        # Calculate Heikin-Ashi
        ha_close = (
            df[f"{symbol}_open"]
            + df[f"{symbol}_high"]
            + df[f"{symbol}_low"]
            + df[f"{symbol}_close"]
        ) / 4

        # HA Open = (Prev HA Open + Prev HA Close) / 2
        # Need to iterate or use shift.
        # For vectorization, we can approximate or use Python loop (slow) or use library.
        # Approximation: HA Open ~ (Open + Close(-1)) / 2 for first pass
        ha_open = (df[f"{symbol}_open"] + df[f"{symbol}_close"].shift(1)) / 2

        # Improved HA Open (Standard recursion)
        # ha_open_list = [ha_open.iloc[0]]
        # for i in range(1, len(df)):
        #     ha_open_list.append((ha_open_list[-1] + ha_close.iloc[i-1]) / 2)
        # ha_open = pd.Series(ha_open_list, index=df.index)

        # Color: Green if HA Close > HA Open
        is_green = ha_close > ha_open

        # ATR Trailing Stop Logic Proxy
        # If Green and Price > ATR Trailing -> Strong Bull
        # We'll just output the "Strength" of the trend: (HA_Close - HA_Open) / ATR
        atr = df[f"{symbol}_atr_14"]
        df[f"{symbol}_alpha_heikin_atr"] = (ha_close - ha_open) / (atr + 1e-9)

        return df

    def _calc_btc_eth_corr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling Correlation between BTC and ETH."""
        corr = df["btc_close"].rolling(24).corr(df["eth_close"])

        # Strategy:
        # High correlation -> Trade together
        # Low correlation -> Idiosyncratic moves (Pair trading opps?)
        # For directional alpha: Maybe "Divergence"?
        # If BTC Up, ETH Down -> Divergence.

        # We'll just use the correlation itself as a feature (Regime filter)
        df["alpha_btc_eth_corr"] = corr

        # Also simple Lead-Lag Alpha: ETH return - BTC return (Relative Strength)
        df["alpha_eth_rel_strength"] = (
            df["eth_close"].pct_change() - df["btc_close"].pct_change()
        )

        return df
