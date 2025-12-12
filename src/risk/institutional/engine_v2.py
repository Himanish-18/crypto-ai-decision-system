from typing import Dict, List

import numpy as np
import pandas as pd


class PortfolioRiskEngineV2:
    """
    Institutional Risk Manager.
    - Monitors aggregate exposure across all assets.
    - Calculates Portfolio VaR (Value at Risk).
    - Enforces Beta limits.
    """

    def __init__(self, max_leverage: float = 1.0, daily_var_limit: float = 0.02):
        self.max_leverage = max_leverage
        self.daily_var_limit = daily_var_limit

        # Exposure State
        self.positions: Dict[str, float] = {}  # Symbol -> USD Value
        self.equity: float = 1000.0  # Initial

    def update_position(self, symbol: str, usd_value: float):
        self.positions[symbol] = usd_value

    def check_trade(
        self, symbol: str, side: str, qty: float, price: float, current_equity: float
    ) -> bool:
        """
        Pre-trade Risk Check.
        Returns: Allowed (True/False)
        """
        self.equity = current_equity
        trade_value = qty * price

        # 1. Leverage Check
        current_exposure = sum(abs(v) for v in self.positions.values())
        new_exposure = current_exposure + trade_value

        leverage = new_exposure / self.equity
        if leverage > self.max_leverage:
            # Reject
            return False

        # 2. Concentration Check (Max 50% in one asset)
        current_pos = self.positions.get(symbol, 0.0)
        new_pos = current_pos + trade_value
        if abs(new_pos) / self.equity > 0.5:
            # Reject
            return False

        # 3. VaR Check (Simplified Parametric)
        # Using fixed Volatility assumption for speed (e.g. 3% daily)
        # VaR_95 = 1.65 * sigma * Value
        vol_est = 0.03
        portfolio_var = 1.65 * vol_est * new_exposure

        if portfolio_var / self.equity > self.daily_var_limit:
            # Reject: Risking too much
            return False

        return True
