from typing import Dict, List

import numpy as np
import pandas as pd


class ScenarioStressTest:
    """
    Institutional Stress Testing Engine.
    Simulates portfolio behavior under historical extreme events.
    """

    SCENARIOS = {
        "COVID_CRASH_2020": {
            "drop_pct": -0.50,
            "vol_spike": 4.0,
            "duration_days": 2,
        },  # 50% drop, 4x vol
        "LUNA_COLLAPSE_2022": {"drop_pct": -0.30, "vol_spike": 3.0, "duration_days": 3},
        "FTX_COLLAPSE_2022": {"drop_pct": -0.25, "vol_spike": 2.5, "duration_days": 5},
        "BLACK_MONDAY_1987": {"drop_pct": -0.22, "vol_spike": 5.0, "duration_days": 1},
    }

    def run_stress_test(
        self, current_portfolio_value: float, current_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply stress factors to current portfolio.
        Returns estimated portfolio value after stress event.
        """
        results = {}

        for name, params in self.SCENARIOS.items():
            # Assume 100% correlation of crypto assets to the stress drop
            # Cash is unaffected.

            crypto_exposure = sum(current_positions.values())

            # Apply drop
            shock_loss = crypto_exposure * params["drop_pct"]

            # Estimate Liquidation Slippage during Vol Spike?
            # Extra loss due to inability to exit.
            # Liquidity Crisis Factor:
            slippage_impact = abs(
                crypto_exposure * 0.05 * params["vol_spike"]
            )  # 5% base slippage scaled

            post_stress_value = current_portfolio_value + shock_loss - slippage_impact
            drawdown = (
                post_stress_value - current_portfolio_value
            ) / current_portfolio_value

            results[name] = {
                "post_stress_value": post_stress_value,
                "drawdown_pct": drawdown,
                "status": "FAIL" if drawdown < -0.30 else "PASS",  # Hard limit 30%
            }

        return results
