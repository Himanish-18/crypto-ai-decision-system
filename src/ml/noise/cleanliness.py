import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("noise_immunity_v3")


class MarketCleanlinessModel:
    """
    v16 Noise Immunity System (Noise v3).
    Quantifies Market 'Chop' vs 'Clean Trend'.
    """

    def __init__(self):
        self.noise_threshold = 0.85

    def analyze_cleanliness(self, df: pd.DataFrame) -> float:
        """
        Returns a 'Noise Probability' score (0.0=Clean, 1.0=Noisy).
        Input: DataFrame with OHLCV.
        """
        if df is None or len(df) < 50:
            return 0.5  # Unknown

        # Feature 1: Efficiency Ratio (Kaufman)
        # ER = Change / SumOfMoves
        # Clean Trend has ER ~ 1.0, Chop ~ 0.0

        change = df["close"].iloc[-1] - df["close"].iloc[-10]
        path = np.sum(np.abs(df["close"].diff().tail(10)))

        if path == 0:
            er = 0.0
        else:
            er = abs(change / path)

        # Feature 2: FFT Noise Ratio (Stub from existing logic)
        # Assume we compute HighFreq/LowFreq energy
        # For prototype: Random logic correlating with small moves

        # Feature 3: Candle Body vs Wick
        # Chop = Small Body, Long Wicks
        body = np.abs(df["close"] - df["open"])
        wick = (df["high"] - df["low"]) - body

        avg_body = body.tail(10).mean()
        avg_wick = wick.tail(10).mean()

        wick_ratio = avg_wick / (avg_body + 1e-9)

        # Synthesis
        # High ER -> Low Noise
        # High Wick Ratio -> High Noise

        noise_score = (1.0 - er) * 0.5 + min(wick_ratio, 1.0) * 0.5

        if noise_score > self.noise_threshold:
            logger.warning(
                f"🌪 Market Noise Detected (Score: {noise_score:.2f}). Blocking Trades."
            )

        return noise_score
