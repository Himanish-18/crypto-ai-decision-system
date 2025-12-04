import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger("risk_engine")

class RiskEngine:
    def __init__(self, account_size: float = 10000.0):
        self.capital = account_size
        # Risk Parameters (Go-Live Safe)
        self.max_risk_per_trade = 0.01  # 1% risk per trade
        self.stop_loss_pct = 0.015      # 1.5% Hard Stop
        self.hard_stop_loss_pct = 0.015 # Alias
        self.trailing_stop_pct = 0.01   # 1% Trailing
        self.take_profit_rr = 2.5       # 2.5R Target
        self.max_position_size_pct = 0.1 # Max 10% of equity per trade
        self.max_capital_per_trade_pct = 0.1 # Alias
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

    def calculate_position_size(self, win_rate: float, entry_price: float, volatility: float = 0.02) -> float:
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
        target_risk_amount = self.capital * self.max_risk_per_trade
        vol_scaled_size_value = target_risk_amount / max(volatility, 0.001) # Avoid div by zero
        
        # 2. Kelly Criterion (Bounded)
        kelly_fraction = (win_rate * 2) - 1
        kelly_size_value = self.capital * max(0, kelly_fraction)
        
        # 3. Combine (Take Minimum)
        # We want the size that satisfies BOTH volatility target and Kelly optimality.
        raw_size_value = min(vol_scaled_size_value, kelly_size_value)
        
        # 4. Hard Limits
        max_allowed_value = self.capital * self.max_position_size_pct
        final_size_value = min(raw_size_value, max_allowed_value)
        
        units = final_size_value / entry_price
        return units

    def check_var_limit(self, current_portfolio_value: float, current_volatility: float, confidence_level: float = 0.95) -> bool:
        """
        Check if Portfolio VaR is within limits.
        VaR = Portfolio_Value * Z_Score * Volatility
        """
        z_score = 1.65 # 95% Confidence
        var = current_portfolio_value * z_score * current_volatility
        
        max_var = self.capital * 0.05 # Max 5% VaR
        
        if var > max_var:
            logger.warning(f"⚠️ VaR Breach! Current: ${var:.2f}, Max: ${max_var:.2f}")
            return False
        return True

    def get_exit_params(self, entry_price: float, volatility: float = 0.015) -> Dict[str, float]:
        """Return stop loss and take profit levels based on volatility."""
        # Dynamic SL based on Volatility (e.g., 2 * Volatility)
        sl_dist = entry_price * (2 * volatility)
        hard_sl = entry_price - sl_dist
        
        # TP based on RR
        rr_ratio = 2.5
        tp = entry_price + (sl_dist * rr_ratio)
        
        return {
            "stop_loss": hard_sl,
            "take_profit": tp,
            "trailing_stop_pct": volatility, # Trail by 1 vol unit
            "soft_exit_prob": self.soft_exit_prob
        }
