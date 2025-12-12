import logging
from typing import Dict, List, Optional

logger = logging.getLogger("portfolio_guard")


class PortfolioGuard:
    """
    Real-time Portfolio Risk Gate.
    """

    def __init__(self, max_leverage: float = 1.0, max_sector_exposure: float = 0.5):
        self.max_leverage = max_leverage
        self.max_sector_exposure = max_sector_exposure

    def check_new_order(
        self, current_exposure: float, new_order_value: float, capital: float
    ) -> bool:
        """
        Check if new order violates portfolio limits.
        """
        total_exposure = current_exposure + new_order_value
        leverage = total_exposure / capital

        if leverage > self.max_leverage:
            logger.warning(
                f"⛔ Risk Veto: Leverage {leverage:.2f} > Max {self.max_leverage}"
            )
            return False

        return True

    def calculate_cvar_gate(
        self, returns: List[float], threshold: float = 0.02
    ) -> bool:
        """
        Check if recent returns indicate high tail risk.
        If cVaR > threshold, halt trading.
        """
        # TODO: Implement rolling cVaR check
        return True
