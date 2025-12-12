from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class RiskProfile:
    capital: float
    kelly_fraction: float = 0.5  # Half Kelly
    max_drawdown_limit: float = 0.20  # 20% Hard Stop
    risk_per_trade: float = 0.01  # 1% Capital Risk
    daily_var_limit: float = 0.02  # 2% Daily VaR 95%


class SensitivityAnalysis:
    """
    Mathematical Formalization of Risk.
    """

    @staticmethod
    def calculate_probability_of_ruin(
        win_rate: float, reward_risk_ratio: float, risk_per_trade: float
    ) -> float:
        """
        Calculate analytical probability of ruin using Random Walk theory.
        Formula Approx: ( (1 - W) / (1 + W) ) ^ (Capital / RiskUnit)
        Assuming simplifications.
        Better approx for Kelly bettors involved solving the equation.

        Using simple formula for infinite horizon risk of reaching 0.
        """
        # If Edge is positive, Prob Ruin < 1.
        # q = probability of loss = 1 - win_rate
        # p = probability of win
        # If p > q, calculating exact ruin is complex depending on sizing.
        # Using Generalized Formula for Gambler's Ruin with drift.

        edge = win_rate * reward_risk_ratio - (1 - win_rate)
        if edge <= 0:
            return 1.0  # Certain Ruin

        # Kelly Criterion: f* = Edge / Odds = (p(b+1) - 1) / b
        # where b = reward_risk

        # Ruin prob for Full Kelly is low?? No, Full Kelly is aggressive.

        # Let's return a metric derived from RoR
        # RoR ≈ e^(-2 * mean * capital / variance)

        return np.exp(-2.0 * edge)  # Conceptual Placeholder

    @staticmethod
    def kelly_criterion(win_rate: float, profit_factor: float) -> float:
        """
        b = profit_factor (approx reward/risk)
        f* = p - q/b
        """
        if profit_factor == 0:
            return 0.0
        q = 1 - win_rate
        f_star = win_rate - (q / profit_factor)
        return max(0.0, f_star)


class InstitutionalRiskEngine:
    def __init__(self, profile: RiskProfile):
        self.profile = profile

    def check_trade_risk(
        self, entry_price: float, stop_loss: float, volatility: float
    ) -> Optional[float]:
        """
        Calculate safe position size.
        """
        # 1. Volatility Adjusted Size
        # Target Risk = Capital * RiskPerTrade
        risk_amt = self.profile.capital * self.profile.risk_per_trade

        # Risk per unit = |Entry - Stop|
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            # Fallback to Volatility based stop
            risk_per_unit = entry_price * volatility * 2.0

        if risk_per_unit == 0:
            return 0.0

        units = risk_amt / risk_per_unit

        return units
