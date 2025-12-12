from typing import Any, Dict, List

import pandas as pd


class ScenarioSimulator:
    """
    v20 Scenario Analysis.
    Simulates portfolio PnL under stress conditions.
    """

    def __init__(self):
        self.standard_scenarios = {
            "crypto_crash_10": {"BTC": -0.10, "ETH": -0.15, "SOL": -0.20},
            "crypto_crash_20": {"BTC": -0.20, "ETH": -0.25, "SOL": -0.30},
            "crypto_meltup_10": {"BTC": 0.10, "ETH": 0.12, "SOL": 0.15},
            "funding_shock": {"rate_shock": -0.05},  # Annualized funding drops 5%
        }

    def simulate(
        self,
        positions: Dict[str, Dict],
        scenario_name: str = None,
        custom_shock: Dict = None,
    ) -> float:
        """
        Calculate Estimated PnL of portfolio under shock.
        positions: Output from PortfolioRiskEngine (dict of asset details)
        """
        shocks = (
            custom_shock
            if custom_shock
            else self.standard_scenarios.get(scenario_name, {})
        )
        if not shocks:
            return 0.0

        pnl = 0.0

        for sym, pos_data in positions.items():
            usd_val = pos_data["usd_value"]

            # Asset Shock
            if sym in shocks:
                shock_pct = shocks[sym]
                pnl += usd_val * shock_pct
            else:
                # Correlated Shock (Beta approximation)
                # If BTC shocks, everything beta-correlated shocks
                if "BTC" in shocks:
                    beta = pos_data.get("beta_mkt", 1.0)
                    shock_pct = shocks["BTC"] * beta
                    pnl += usd_val * shock_pct

        return pnl

    def run_all(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        results = {}
        for name in self.standard_scenarios:
            results[name] = self.simulate(positions, scenario_name=name)
        return results
