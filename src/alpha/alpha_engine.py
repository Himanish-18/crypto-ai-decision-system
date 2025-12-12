import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("alpha_engine")


class AlphaEngine:
    """
    Multi-Source Alpha Engine.
    Computes alpha features from On-Chain data, Orderbook Microstructure, and Volatility Regimes.
    """

    def __init__(self):
        pass

    def get_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point to compute all alphas and append to the dataframe.
        """
        logger.info("🚀 Computing Alpha Features...")
        df = df.copy()

        # 1. On-Chain Alphas (Whale, Exchange, Miner)
        df = self._compute_on_chain_alphas(df)

        # 2. Microstructure Alphas (Spread, OFI, VPIN)
        df = self._compute_microstructure_alphas(df)

        # 3. Volatility Regime
        df = self._compute_volatility_regime(df)

        # 4. Composite Alpha Score (Simple equal weight or learned weight)
        # For now, let's just return the raw alphas.
        # The strategy optimizer will blend them.

        return df

    def _compute_on_chain_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate/Compute On-Chain Metrics.
        In a real scenario, this would fetch from Glassnode/CryptoQuant.
        Here we will use price/volume proxies to simulate the signal behavior.
        """
        # A) Whale Flow Proxy: Large volume spikes relative to recent average
        # Logic: If Volume > 2 * MovingAvg(Volume), assume Whale Activity.
        # Direction: If Close > Open (Green Candle) -> Inflow, else Outflow.

        vol_ma = df["btc_volume"].rolling(24).mean()
        vol_std = df["btc_volume"].rolling(24).std()

        # Z-score of volume
        vol_z = (df["btc_volume"] - vol_ma) / (vol_std + 1e-9)

        # Whale Flow: +1 if high vol up, -1 if high vol down, else 0
        # We smooth it to make it a continuous signal
        df["alpha_whale_flow"] = np.where(
            vol_z > 2, np.sign(df["btc_close"] - df["btc_open"]), 0
        )
        # Decay the signal
        df["alpha_whale_flow"] = df["alpha_whale_flow"].ewm(span=12).mean()

        # B) Exchange Balance Delta Proxy
        # Logic: High Volatility + Downside -> Inflow to Exchange (Sell pressure)
        # Low Volatility + Upside -> Outflow from Exchange (Accumulation)
        # We'll use a contrarian indicator based on RSI and Volatility
        rsi = df["btc_rsi_14"]

        # If RSI is high (>70) and price is up, assume potential distribution (inflow to exchange) -> Negative Alpha
        # If RSI is low (<30) and price is down, assume potential accumulation (outflow from exchange) -> Positive Alpha

        # Normalize RSI to -1 to 1
        rsi_norm = (rsi - 50) / 50

        # Exchange Delta Proxy: Inverse of RSI trend?
        # Let's define it as: "Net Flow to Exchange". Positive = Bad for price.
        # So Alpha should be Negative of Net Flow.
        # We'll approximate "Net Flow" as proportional to RSI (Overbought = Inflow, Oversold = Outflow)
        df["alpha_exchange_delta"] = (
            -rsi_norm
        )  # High RSI -> Negative Alpha (Expect Reversal)

        # C) Miner Reserve Flow Proxy
        # Logic: Miners sell into strength.
        # If Price is making new highs but momentum is waning -> Miner Selling.
        # We'll use MACD Histogram as a proxy for "Momentum Strength".
        macd_hist = df["btc_macd_diff"]
        df["alpha_miner_flow"] = (
            macd_hist  # Positive Momentum -> Positive Alpha (Miners holding)
        )

        return df

    def _compute_microstructure_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Orderbook/Microstructure Alphas.
        Using High/Low/Close/Volume as proxies for tick data features.
        """
        # E) Orderbook Microstructure Alpha

        # 1. Spread Proxy (Corwin-Schultz or similar, but simpler: High - Low)
        # Relative Spread = (High - Low) / Close
        # High Spread -> Low Liquidity -> Higher Risk/Volatility
        df["alpha_spread"] = (df["btc_high"] - df["btc_low"]) / df["btc_close"]

        # 2. Order Flow Imbalance (OFI) Proxy
        # OFI usually requires L2 data.
        # Proxy: (Close - Open) / (High - Low) -> "Close Location Value" (CLV)
        # CLV ~ 1 -> Buying Pressure, CLV ~ -1 -> Selling Pressure
        df["alpha_ofi"] = (2 * df["btc_close"] - df["btc_high"] - df["btc_low"]) / (
            df["btc_high"] - df["btc_low"] + 1e-9
        )

        # 3. VPIN-like Toxic Order Flow Proxy
        # Volume-Synchronized Probability of Informed Trading
        # Proxy: Volume * |Ret| / Total Volume (Flow Toxicity)
        # High toxicity -> Reversal likely? Or Continuation?
        # Usually High VPIN -> Crash risk.
        # We'll compute "Trade Intensity" = Volume / Time (Time is constant 1H)
        # And "Price Impact" = |Return| / Volume (Kyle's Lambda proxy)

        ret = df["btc_close"].pct_change().abs()
        # Amihud Illiquidity Measure = |Ret| / Volume
        # High Illiquidity -> Price is sensitive to volume.
        df["alpha_vpin_proxy"] = ret / (df["btc_volume"] + 1e-9)
        # Invert it? Or use it as a regime filter?
        # Let's use it as a feature: "Market Impact Cost"
        # High Impact -> Negative Alpha (Fragile market)
        df["alpha_vpin_proxy"] = -df[
            "alpha_vpin_proxy"
        ]  # Higher is better (more liquid)

        return df

    def _compute_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Realized Volatility Regime Labels.
        """
        # D) Realized Volatility
        # 24H Rolling Std Dev of Returns
        ret = df["btc_close"].pct_change()
        realized_vol = ret.rolling(24).std()

        # Normalize Volatility (Z-Score over longer window, e.g., 30 days = 720 hours)
        long_term_vol_mean = realized_vol.rolling(720).mean()
        long_term_vol_std = realized_vol.rolling(720).std()

        vol_z = (realized_vol - long_term_vol_mean) / (long_term_vol_std + 1e-9)

        # Regime Labels:
        # Low Vol: Z < -1
        # Normal: -1 <= Z <= 1
        # High Vol: Z > 1

        # We'll output a continuous "Regime Score" where Higher = Higher Volatility
        # But for Alpha, we might want "Stability".
        # Let's just output the Z-score as a feature.
        df["alpha_regime_vol"] = vol_z

        # We can also create one-hot or categorical if needed, but trees handle continuous well.

        return df
