import logging
import random
from typing import Dict, List

logger = logging.getLogger("market_router_v2")


class MarketRouterV2:
    """
    v16 Autonomous Market Switcher.
    Scans [BTC, ETH, SOL, BNB] + Perps.
    Selects Best Market based on Liquidity & Volatility Score.
    """

    def __init__(self):
        self.markets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
        self.current_focus = "BTC/USDT"

    def scan_markets(self, market_data_feed: Dict[str, Dict]) -> str:
        """
        Evaluate all markets and return the best ticker to focus on.
        """
        scores = []

        for symbol in self.markets:
            data = market_data_feed.get(symbol, {})
            if not data:
                continue

            # Metric 1: Liquidity (Volume / Spread)
            # Stub: Random simulation for v16 proto
            liquidity_score = random.uniform(0.7, 1.0)

            # Metric 2: Trends (Adx / Momentum)
            trend_score = random.uniform(0.0, 1.0)

            # Metric 3: Volatility (ATR)
            # We want Volatility but not Noise
            vol_score = random.uniform(0.5, 1.0)

            # Composite Score
            # Heavy weighting on Trend & Liquidity
            total_score = (
                (0.4 * liquidity_score) + (0.4 * trend_score) + (0.2 * vol_score)
            )

            scores.append((symbol, total_score))

        if not scores:
            return self.current_focus

        # Select Winner
        winner = max(scores, key=lambda x: x[1])
        best_sym = winner[0]

        if best_sym != self.current_focus:
            logger.info(
                f"🔭 Market Focus Switch: {self.current_focus} -> {best_sym} (Score: {winner[1]:.2f})"
            )
            self.current_focus = best_sym

        return self.current_focus
