import logging
from typing import Dict, List

logger = logging.getLogger("exposure_report")


class ExposureReport:
    """
    Generates real-time risk exposure reports.
    """

    @staticmethod
    def calculate_leverage(positions_value: float, equity: float) -> float:
        if equity == 0:
            return 0.0
        return positions_value / equity

    @staticmethod
    def calculate_metrics(
        portfolio: Dict[str, float], prices: Dict[str, float], equity: float
    ) -> Dict[str, float]:
        """
        Compute Greeks & Exposure.
        """
        total_exposure = 0.0
        for symbol, qty in portfolio.items():
            price = prices.get(symbol, 0.0)
            total_exposure += abs(qty * price)

        leverage = ExposureReport.calculate_leverage(total_exposure, equity)

        # Delta/Gamma Placeholders (Linear assets -> Delta=1, Gamma=0)

        return {
            "total_exposure_usd": total_exposure,
            "net_leverage": leverage,
            "margin_utilization": leverage / 5.0,  # Assuming max 5x
            "delta_exposure": total_exposure,  # Long Only assumption
            "gamma_exposure": 0.0,
        }
