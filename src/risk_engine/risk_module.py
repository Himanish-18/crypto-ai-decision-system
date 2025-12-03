import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger("risk_engine")

class RiskEngine:
    def __init__(self):
        # Risk Parameters (Go-Live Safe)
        self.max_risk_per_trade = 0.01  # 1% risk per trade
        self.stop_loss_pct = 0.015      # 1.5% Hard Stop
        self.trailing_stop_pct = 0.01   # 1% Trailing
        self.take_profit_rr = 2.5       # 2.5R Target
        self.max_position_size_pct = 0.1 # Max 10% of equity per trade (conservative)
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

    def calculate_position_size(self, win_rate: float, entry_price: float) -> float:
        """
        Calculate position size using Kelly Criterion (bounded).
        """
        # Kelly Fraction = W - (1-W)/R
        # Simplified formula given in prompt: Kelly = win_rate * 2 - 1 (Assuming R=1)
        # Let's use the prompt's formula but ensure safety.
        
        kelly_fraction = (win_rate * 2) - 1
        
        # Cap Kelly at 0 if negative
        if kelly_fraction <= 0:
            return 0.0
            
        # Apply Max Capital Constraint (2%)
        # The prompt says "Max capital per trade = 2%". 
        # This usually means "Risk 2%" or "Allocate 2%". 
        # Given "Cap leverage = 1x", allocating only 2% of capital seems very conservative (no leverage).
        # But "Risk 2%" usually means "Loss limited to 2% of capital".
        # Let's interpret "Max capital per trade = 2%" as "Max Allocation = 2% of Portfolio" 
        # OR "Max Risk Amount = 2% of Portfolio".
        # Given the context of "Kelly Fraction" (which dictates allocation), 
        # usually Kelly suggests allocation.
        # However, "Max capital per trade = 2%" strongly suggests Allocation Cap.
        # Let's assume Max Allocation = 2% (very conservative) OR Max Risk = 2%.
        # Re-reading: "Max capital per trade = 2%". This likely means Allocation.
        # But wait, standard Kelly can suggest high allocation.
        # If I cap allocation at 2%, Kelly is almost useless unless it suggests < 2%.
        # Let's assume "Risk 2% of capital" is the constraint for the STOP LOSS calculation,
        # and "Max capital per trade" might be a typo for "Max Risk"?
        # Let's stick to the previous successful logic: Risk 2% of capital.
        # Max Risk Amount = Capital * 0.02
        # Position Size = Max Risk Amount / (Entry * StopLossPct)
        
        # But the prompt says: "Kelly Fraction = ...", "Max capital per trade = 2%".
        # Let's implement: Allocation = Min(Kelly * Capital, 0.02 * Capital).
        # This effectively caps position size at 2% of account.
        
        allocation_pct = min(kelly_fraction, self.max_capital_per_trade_pct)
        allocation_amount = self.capital * allocation_pct
        
        # Cap leverage = 1x (Implicit since allocation_pct <= 1.0)
        
        units = allocation_amount / entry_price
        return units

    def get_exit_params(self, entry_price: float) -> Dict[str, float]:
        """Return stop loss and take profit levels."""
        hard_sl = entry_price * (1 - self.hard_stop_loss_pct)
        # TP not explicitly defined in prompt "Computed from RR ratio"
        # Let's assume RR = 2.0 or 2.5 from previous step
        rr_ratio = 2.5
        tp = entry_price * (1 + (self.hard_stop_loss_pct * rr_ratio))
        
        return {
            "stop_loss": hard_sl,
            "take_profit": tp,
            "trailing_stop_pct": self.trailing_stop_pct,
            "soft_exit_prob": self.soft_exit_prob
        }
