import logging

import numpy as np

logger = logging.getLogger("execution.impact")


class MarketImpactModel:
    """
    v24 Square Root Law Implementation.
    Impact = Y * Volatility * sqrt(OrderSize / Volume)
    """

    def __init__(self, y_coeff: float = 0.5):
        self.y = y_coeff

    def estimate_impact_bps(
        self, size_usd: float, daily_vol_usd: float, volatility: float
    ) -> float:
        """
        Returns expected price impact in basis points.
        """
        # Linear approximation of sqrt law for BPS
        # Impact ~= Y * sigma * sqrt(Q / V)

        fraction = size_usd / (daily_vol_usd + 1e-9)
        impact = self.y * volatility * np.sqrt(fraction)

        return impact * 10000  # Convert to BPS

    def recommend_max_size(
        self, max_impact_bps: float, daily_vol: float, volatility: float
    ) -> float:
        """
        Reverse solve for Max Size given Slippage Tolerence.
        """
        # I = Y * sigma * sqrt(Q / V)
        # I / (Y * sigma) = sqrt(Q / V)
        # (I / (Y * sigma))^2 = Q / V
        # Q = V * (I / (Y * sigma))^2

        impact = max_impact_bps / 10000
        term = impact / (self.y * volatility + 1e-9)
        max_q = daily_vol * (term**2)

        return max_q
