import numpy as np


class SlippageModelV2:
    """
    Advanced Slippage Models (v2) for Institutional Backtesting.
    """

    @staticmethod
    def spread_based_slippage(price: float, spread_bps: float = 5.0) -> float:
        """
        Slippage = Half Spread.
        """
        return price * (spread_bps / 10000.0) * 0.5

    @staticmethod
    def volume_weighted_slippage(
        price: float, order_size: float, volume_profile: float, volatility: float
    ) -> float:
        """
        Slippage increases as Order Size consumes more of available Volume.
        Linear Impact approximation + Volatility Penalty.
        """
        if volume_profile == 0:
            return price * 0.01  # Fallback 1%

        participation = order_size / volume_profile
        base_slip = price * 0.0005  # 5bps base

        impact = 10.0 * participation**2  # Quadratic penalty for large size
        vol_penalty = volatility * 100.0

        return base_slip * (1 + impact + vol_penalty)
