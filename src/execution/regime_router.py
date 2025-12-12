import logging
from typing import Any, Dict

# Setup Logging
logger = logging.getLogger("regime_router")


class RegimeRouter:
    """
    Routes execution strategy and risk parameters based on Market Regime.

    Regimes:
    0: Chop / Mean Reversion -> Passive Entry, Tight Stops.
    1: Trend (Bull/Bear) -> Aggressive/Passive Entry, Wide Stops.
    2: Volatility Spike -> Suspend / Min Size.
    """

    def __init__(self):
        pass

    def get_execution_config(self, regime: int) -> Dict[str, Any]:
        """
        Get execution parameters for the given regime.
        """
        config = {
            "style": "PASSIVE",
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.0,
            "size_multiplier": 1.0,
            "can_trade": True,
        }

        if regime == 0:  # Chop
            logger.info("🌀 Regime: Chop. Using Mean Reversion Logic.")
            config.update(
                {
                    "style": "PASSIVE",  # Strict Maker
                    "stop_loss_multiplier": 0.8,  # Tight SL
                    "take_profit_multiplier": 1.2,  # Quick TP
                    "size_multiplier": 0.8,  # Reduced Size
                }
            )

        elif regime == 1:  # Trend
            logger.info("🚀 Regime: Trend. Using Momentum Logic.")
            config.update(
                {
                    "style": "PASSIVE",  # Prefer Maker but can be aggressive if needed
                    "stop_loss_multiplier": 1.5,  # Wide SL to ride noise
                    "take_profit_multiplier": 3.0,  # Let winners run
                    "size_multiplier": 1.0,  # Full Size
                }
            )

        elif regime == 2:  # Volatility / Spike
            logger.warning("⚠️ Regime: High Volatility. Suspending/Reducing.")
            config.update(
                {
                    "style": "PASSIVE",
                    "size_multiplier": 0.0,  # Suspend
                    "can_trade": False,
                }
            )

        return config


# Mock for Verification
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    router = RegimeRouter()

    for r in [0, 1, 2]:
        config = router.get_execution_config(r)
        logger.info(f"Regime {r} Config: {config}")
