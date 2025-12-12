import logging
from typing import Dict, List

import pandas as pd

from src.execution.hft.router_multi_exchange import CrossExchangeRouter
from src.risk.global_risk_guard import GlobalRiskGuard

logger = logging.getLogger("portfolio_v15")


class PortfolioCoordinator:
    """
    v15 Engine.
    Orchestrates Multi-Asset Trading.
    Assets: BTC, ETH, SOL, AVAX, BNB.
    """

    def __init__(self):
        self.assets = ["BTC", "ETH", "SOL", "AVAX", "BNB"]
        self.risk_guard = GlobalRiskGuard()
        self.router = CrossExchangeRouter()

        # State
        self.allocations = {sym: 0.20 for sym in self.assets}  # Equal weight init
        self.positions = {sym: 0.0 for sym in self.assets}
        self.total_equity = 10000.0

    def on_tick(self, market_data: Dict[str, Dict]):
        """
        Main Loop for Portfolio.
        market_data: { "BTC": {price, ob...}, "ETH": ... }
        """
        # 1. Update Global Risk State
        # Calculate Equity
        current_eq = self.total_equity  # Placeholder, should sum(pos * price)
        self.risk_guard.update_equity(current_eq)

        if not self.risk_guard.can_trade():
            logger.warning("⛔ Global Risk Guard Active. Halting.")
            return

        # 2. Iterate Assets
        for sym in self.assets:
            data = market_data.get(sym)
            if not data:
                continue

            # 3. Generate Signal (Stub for Multi-Asset, usually calls 5x LiveSignalEngines)
            # For v15 prototype, we use random/momentum stub
            score = 0.5  # Neutral

            # 4. Allocation Logic (PPO Wrapper)
            # If Score > 0.7 -> Alloc += 5%
            target_alloc = self.allocations[sym]

            # 5. Execution
            # If alloc diff > threshold, rebalance via Router
            # self.router.get_best_venue(...)
            pass

    def rebalance(self):
        logger.info("⚖️ Portfolio Rebalance Triggered")
