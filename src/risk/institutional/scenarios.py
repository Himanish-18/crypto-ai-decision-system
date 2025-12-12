import logging
from typing import Dict, List


class ScenarioSimulator:
    """
    Stress-tests the portfolio against predefined market shocks.
    """

    def __init__(self):
        self.scenarios = {
            "FLASH_CRASH": {"btc": -0.05, "eth": -0.06, "corr": 1.0},
            "FED_HIKE": {"btc": -0.02, "eth": -0.025, "corr": 0.9},
            "STABLE_COIN_DEPEG": {"btc": -0.10, "eth": -0.15, "corr": 0.5},
            "LIQUIDITY_CRUNCH": {"btc": -0.01, "eth": -0.01, "spread_mult": 5.0},
        }

    def simulate(
        self, positions: Dict[str, float], scenario_name: str
    ) -> Dict[str, float]:
        """
        Returns estimated PnL impact and Margin Utilization.
        """
        scenario = self.scenarios.get(scenario_name)
        if not scenario:
            return {}

        total_pnl = 0.0

        for symbol, size_usd in positions.items():
            sym_key = symbol.lower().split("usdt")[0]
            shock = scenario.get(sym_key, -0.05)  # Default shock

            pnl_impact = size_usd * shock
            total_pnl += pnl_impact

        return {"impact_usd": total_pnl, "scenario": scenario_name}
