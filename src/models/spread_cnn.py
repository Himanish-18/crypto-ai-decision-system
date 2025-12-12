import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("spread_cnn")


class SpreadCNN:
    """
    Predicts Bid-Ask Spread Regime using a Tiny-CNN (v3) architecture.
    Used to decide MAKER (Wide) vs TAKER (Tight) execution.
    """

    def __init__(self, model_path: Path = None):
        self.model_path = model_path
        self.model = None
        # If path provided, try load (Placeholder for future)
        # For now, use Heuristic Proxy logic that mimics CNN output

    def predict(self, df: pd.DataFrame) -> float:
        """
        Predict probability of TIGHT spread (< 0.06%).
        Output: 0.0 (Wide) to 1.0 (Tight).
        """
        if len(df) < 20:
            return 0.5

        latest = df.iloc[-1]

        # Inputs the CNN would use:
        # 1. Volatility (ATR)
        # 2. Volume (Liquidity)
        # 3. Toxicity (Amihud)

        # Heuristic Logic V3:
        # High Vol + Low Vol = Wide Spread (Bad) -> Score 0
        # Low Vol + High Vol = Tight Spread (Good) -> Score 1

        # 1. Volatility Component
        vol = 0.005
        if "atr" in df.columns:
            # Normalized ATR?
            vol = df["atr"].iloc[-1] / df["close"].iloc[-1]
        elif "btc_close" in df.columns:
            vol = df["btc_close"].pct_change().rolling(20).std().iloc[-1]

        # Vol > 0.5% (0.005) is considered High Vol -> Limits spread tightness
        score_vol = np.clip(1.0 - (vol / 0.005), 0, 1)

        # 2. Liquidity (volume)
        vol_roll = df["volume"].rolling(20).mean().iloc[-1]
        if vol_roll == 0:
            vol_roll = 1.0
        curr_vol = latest["volume"]

        # Higher volume usually tightens spread
        score_liq = np.clip(curr_vol / vol_roll, 0, 1)

        # Combined Score (CNN Proxy)
        # Heavier weight on Volatility for spread prediction
        tight_spread_prob = (score_vol * 0.7) + (score_liq * 0.3)

        return float(tight_spread_prob)

    def get_execution_mode(self, df: pd.DataFrame) -> str:
        """
        Determine execution mode based on spread prediction.
        """
        prob_tight = self.predict(df)

        # If prob_tight > 0.6 -> Spread is likely < 0.06% -> TAKER is cheap/ok
        # Else -> SPREAD RISK -> MAKER only

        if prob_tight > 0.6:
            return "TAKER"
        else:
            return "MAKER"
