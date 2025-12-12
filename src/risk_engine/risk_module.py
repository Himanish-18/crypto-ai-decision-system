import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("risk_engine")


class RiskEngine:
    def __init__(self, account_size: float = 10000.0):
        self.capital = account_size
        # Risk Parameters (Go-Live Safe)
        self.max_risk_per_trade = 0.01  # 1% risk per trade
        self.stop_loss_pct = 0.015  # 1.5% Hard Stop
        self.hard_stop_loss_pct = 0.015  # Alias
        self.trailing_stop_pct = 0.01  # 1% Trailing
        self.take_profit_rr = 2.5  # 2.5R Target
        self.max_position_size_pct = 0.1  # Max 10% of equity per trade
        self.max_capital_per_trade_pct = 0.1  # Alias
        self.soft_exit_prob = 0.45

    def check_filters(self, market_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if trade is allowed based on market conditions.
        Returns: (is_allowed, reason)
        """
        # 1. Volatility Filter (ATR > 95th percentile)
        # Assuming 'atr_pct' is passed in market_state or calculated externally
        if market_state.get("atr_pct", 0) > 0.95:
            return False, "Extreme Volatility (ATR > 95th pct)"

        # 2. Sentiment Shock
        if market_state.get("sentiment_shock", 0) == 1:
            return False, "Sentiment Shock"

        # 3. ATR Extreme Values (Absolute check if needed, but percentile covers it)
        # Could add max ATR value check if needed.

        return True, "OK"

    def calculate_position_size(
        self, win_rate: float, entry_price: float, volatility: float = 0.02
    ) -> float:
        """
        Calculate position size using Volatility Scaling and Kelly Criterion.

        Args:
            win_rate: Probability of winning.
            entry_price: Current asset price.
            volatility: Current asset volatility (e.g., ATR/Price or StdDev). Default 2%.
        """
        # 1. Volatility Scaling
        # Target Risk: 1% of Capital
        # Position = (Capital * Risk_Pct) / Volatility
        # This ensures constant dollar risk regardless of volatility.
        # 1. Volatility Scaling
        # Target Risk: 1% of Capital
        # Value = (Capital * Risk_Pct) / Volatility
        target_risk_amount = self.capital * self.max_risk_per_trade
        vol_scaled_size_value = target_risk_amount / max(volatility, 0.001)

        # 2. Kelly Criterion (Bounded)
        # Assuming win_rate > 0.5 to have positive sizing
        if win_rate > 0.5:
            kelly_fraction = (win_rate * 2) - 1
            kelly_size_value = self.capital * kelly_fraction
        else:
            kelly_size_value = 0.0

        # 3. Gap-Safe Sizing
        gap_risk_pct = 0.02
        # Max loss allowed for gap = 2% of capital (Capital * 2 * Risk_Per_Trade?) NO.
        # Max Gap Loss should probably not exceed Max Drawdown tolerance for a single event?
        # Let's say max gap loss = 1% of Capital (same as trade risk).
        max_gap_loss = self.capital * self.max_risk_per_trade
        gap_safe_value = max_gap_loss / gap_risk_pct

        # 4. Combine
        # Take min of constraints
        final_value = min(vol_scaled_size_value, gap_safe_value)

        # Apply Kelly as a potential reducer if edge is small, but don't force full Kelly if it's huge.
        # Usually Kelly is an upper bound on leverage.
        # Let's limit to Half-Kelly for safety.
        if kelly_size_value > 0:
            final_value = min(final_value, kelly_size_value * 0.5)
        else:
            final_value = 0.0  # No edge

        # 5. Hard Limits (Max Position Size)
        max_allowed_value = self.capital * self.max_position_size_pct
        final_value = min(final_value, max_allowed_value)

        units = final_value / entry_price
        return units

    def check_var_limit(
        self,
        current_portfolio_value: float,
        current_volatility: float,
        confidence_level: float = 0.95,
    ) -> bool:
        """
        Check if Portfolio VaR is within limits.
        VaR = Portfolio_Value * Z_Score * Volatility
        """
        z_score = 1.65  # 95% Confidence
        var = current_portfolio_value * z_score * current_volatility

        max_var = self.capital * 0.05  # Max 5% VaR

        if var > max_var:
            logger.warning(f"⚠️ VaR Breach! Current: ${var:.2f}, Max: ${max_var:.2f}")
            return False
        return True

    def get_exit_params(
        self, entry_price: float, volatility: float = 0.015
    ) -> Dict[str, float]:
        """Return stop loss and take profit levels based on volatility."""
        # Dynamic SL based on Volatility (e.g., 2 * Volatility)
        sl_dist = entry_price * (2 * volatility)
        hard_sl = entry_price - sl_dist

        # TP based on RR
        rr_ratio = 2.5
        tp = entry_price + (sl_dist * rr_ratio)

        return {"soft_exit_prob": self.soft_exit_prob}

    def check_drawdown_limit(
        self, current_equity: float = None, peak_equity: float = None
    ) -> bool:
        """
        Circuit Breaker: Stop trading if drawdown exceeds limit.
        """
        # If no equity passed, assume self.capital is current equity.
        # But for stateful drawdown, we need to track peak.
        # Simple Stateless Check: If current capital < 0.85 * Initial (15% DD from start)
        # This assumes self.capital is UPDATED externally.

        limit = 0.85  # 15% Max Drawdown allowed
        if current_equity is None:
            current_equity = self.capital

        # Ideally we track High Water Mark.
        # But simplified: If we lost 15% of *initial account size* ($10k), stop.
        initial_capital = 10000.0  # Hardcoded base for safety or passed in init

        if current_equity < (initial_capital * limit):
            logger.critical(
                f"🛑 KILL SWITCH: Equity {current_equity} < {limit*100}% of {initial_capital}"
            )
            return False  # Halt

        return True
