import logging
from typing import Dict, List

# Setup Logging
logger = logging.getLogger("portfolio_allocator")


class PortfolioAllocator:
    """
    Allocates capital across multiple symbols based on weights and risk limits.

    Features:
    - Fractional allocation (e.g., 40% BTC, 30% ETH, 30% BNB).
    - Hard exposure limits per asset.
    - Global leverage cap.
    """

    def __init__(self, total_capital: float = 10000.0):
        self.total_capital = total_capital
        self.allocations: Dict[str, float] = {}  # Symbol -> Allocation Amount
        self.max_leverage = 1.0  # No leverage for Alpha 4

        # Default Weights (can be dynamic)
        self.target_weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.3, "BNB/USDT": 0.2}

    def set_target_weights(self, weights: Dict[str, float]):
        """Update target portfolio weights."""
        if not abs(sum(weights.values()) - 1.0) < 1e-6:
            logger.warning(
                f"Weights do not sum to 1.0: {sum(weights.values())}. Normalizing..."
            )
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        self.target_weights = weights
        logger.info(f"Updated Target Weights: {self.target_weights}")

    def get_allocation(self, symbol: str) -> float:
        """Get the max allowed capital allocation for a symbol."""
        weight = self.target_weights.get(symbol, 0.0)
        allocation = self.total_capital * weight * self.max_leverage
        return allocation

    def check_exposure_limit(
        self, symbol: str, current_exposure: float, new_trade_size: float
    ) -> bool:
        """
        Check if a new trade would breach exposure limits.
        """
        max_alloc = self.get_allocation(symbol)
        if current_exposure + new_trade_size > max_alloc:
            logger.warning(
                f"🚫 Exposure Limit Breach for {symbol}. Max: {max_alloc}, Current: {current_exposure}, New: {new_trade_size}"
            )
            return False
        return True


# Mock for Verification
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    allocator = PortfolioAllocator(total_capital=10000)

    # Check BTC Allocation
    btc_alloc = allocator.get_allocation("BTC/USDT")
    logger.info(f"BTC Allocation: ${btc_alloc:.2f}")

    # Check Limit
    is_ok = allocator.check_exposure_limit(
        "BTC/USDT", current_exposure=4000, new_trade_size=1500
    )
    logger.info(f"Trade Allowed (4000+1500 <= 5000)? {is_ok}")

    is_ok = allocator.check_exposure_limit(
        "BTC/USDT", current_exposure=4000, new_trade_size=500
    )
    logger.info(f"Trade Allowed (4000+500 <= 5000)? {is_ok}")
