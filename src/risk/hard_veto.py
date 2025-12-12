import logging
from typing import Dict

from src.risk.stress_grid import StressGrid
from src.risk.var_engine import VarEngine

logger = logging.getLogger("risk.hard_veto")


class HardVeto:
    """
    v24 Terminal Risk Layer.
    Can FORCE LIQUIDATION or BLOCK ALL TRADES.
    """

    def __init__(self):
        self.stress_engine = StressGrid()

    def check_portfolio_health(
        self, nav: float, positions: Dict[str, float], metrics: Dict[str, float]
    ) -> bool:
        """
        Returns False if portfolio is in critical danger.
        """
        # 1. Check Max Drawdown
        current_dd = metrics.get("drawdown_pct", 0.0)
        if current_dd < -0.15:
            logger.critical("🛑 HARD VETO: Max Drawdown Breach (-15%). Halting.")
            return False

        # 2. Check Liquidity Regime (from external source passed in metrics)
        regime = metrics.get("regime", "NEUTRAL")
        if regime == "LIQ_CRUNCH":
            logger.critical("🛑 HARD VETO: Liquidity Crunch Detected. Halting.")
            return False

        # 3. Run Stress Test
        # Stub weights (equal weight)
        weights = {k: 1.0 / len(positions) for k in positions} if positions else {}
        if not weights:
            return True  # Empty portfolio is safe

        stress_df = self.stress_engine.run_stress_test(weights, nav)
        fails = stress_df[stress_df["Survival_Status"] == "FAIL"]

        # If > 50% of scenarios kill us, stop.
        if len(fails) > (len(stress_df) * 0.5):
            logger.critical(
                "🛑 HARD VETO: Portfolio too fragile (Failed >50% Stress Scenarios)."
            )
            return False

        return True
