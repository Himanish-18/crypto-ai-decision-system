import logging

import pandas as pd

logger = logging.getLogger("hedge_manager")


class HedgeManager:
    """
    Manages Cross-Asset Hedging (e.g., BTC Long + ETH Short).
    Goal: Reduce Beta Exposure during High Risk regimes.
    """

    def __init__(self, target_beta=0.0):
        self.target_beta = target_beta
        self.active_hedges = {}  # {symbol: size}

    def calculate_hedge_ratio(self, prices_a: pd.Series, prices_b: pd.Series) -> float:
        """
        Calculates Beta of Asset A vs Asset B (Benchmark).
        Beta = Cov(A, B) / Var(B)
        """
        if len(prices_a) != len(prices_b):
            min_len = min(len(prices_a), len(prices_b))
            prices_a = prices_a.iloc[-min_len:]
            prices_b = prices_b.iloc[-min_len:]

        ret_a = prices_a.pct_change().dropna()
        ret_b = prices_b.pct_change().dropna()

        cov = ret_a.cov(ret_b)
        var = ret_b.var()

        if var == 0:
            return 1.0

        beta = cov / var
        return beta

    def check_hedge_requirement(
        self,
        portfolio_exposure: float,
        market_regime: str,
        btc_prices: list,
        eth_prices: list,
        iv_critical: bool = False,
    ) -> dict:
        """
        Determines if a hedge is needed.
        Returns: {"action": "OPEN_HEDGE", "symbol": "ETH/USDT", "size": 0.5} or None
        """
        # v12 IV Logic: If IV Critical, force hedge to neutralize delta
        if iv_critical:
            logger.warning(
                "🛡️ Hedge Manager: CRITICAL IV DETECTED. Forcing Delta Neutrality."
            )
            # If we have Long Exposure, Short BTC Perp to flatten.
            if portfolio_exposure > 0:
                return {
                    "action": "HEDGE_SHORT_BTC",
                    "ratio": 1.0,  # Full Hedge (Delta Neutral)
                }

        if market_regime != "CRASH_RISK" and market_regime != "VOLATILE":
            # No hedge needed in stable markets for this strategy
            return None

        if portfolio_exposure <= 0:
            return None  # No long exposure to hedge

        # Calculate Beta (ETH vs BTC)
        try:
            s_btc = pd.Series(btc_prices)
            s_eth = pd.Series(eth_prices)
            beta = self.calculate_hedge_ratio(
                s_eth, s_btc
            )  # Beta of ETH relative to BTC

            logger.info(
                f"🛡️ Hedge Manager: Regime {market_regime}. Beta(ETH|BTC): {beta:.2f}"
            )

            return {"action": "HEDGE_SHORT_ETH", "ratio": 0.5}  # Hedge 50% of exposure
        except Exception as e:
            logger.error(f"Hedge calc failed: {e}")
            return None
