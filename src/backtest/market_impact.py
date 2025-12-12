import numpy as np


class MarketImpactModel:
    """
    Market Impact Estimation using Square-Root Law.
    I = Y * sigma * sqrt(Q / V)

    Where:
    I = Price Impact (absolute)
    Y = Constant (usually 0.7 to 1.0)
    sigma = Daily Volatility (absolute or proportional?) -> usually price * daily_vol
    Q = Order Size (units)
    V = Daily Volume (units)
    """

    def __init__(self, Y: float = 0.8):
        self.Y = Y

    def estimate_impact(
        self,
        price: float,
        daily_vol_pct: float,
        order_qty: float,
        daily_volume_qty: float,
    ) -> float:
        """
        Calculate expected price impact.
        Returns absolute price change expected.
        """
        if daily_volume_qty == 0:
            return 0.0

        participation = order_qty / daily_volume_qty

        # Impact in price units
        # I = Y * (Price * Vol) * sqrt(Part)
        impact = self.Y * (price * daily_vol_pct) * np.sqrt(participation)

        return impact
