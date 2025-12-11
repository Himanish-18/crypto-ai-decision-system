import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger("global_risk_guard")

class GlobalRiskGuard:
    """
    v15 Global Risk Overwatch.
    - Monitors Portfolio Drawdown (Stop Trading if < -3%).
    - Detects Correlation Shocks.
    """
    def __init__(self, max_drawdown_pct: float = 0.03):
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.circuit_breaker_active = False
        self.liquidate_all_signal = False
        
    def update_equity(self, total_equity_usd: float):
        """
        Update equity high-water mark and check drawdown.
        """
        self.current_equity = total_equity_usd
        if total_equity_usd > self.peak_equity:
            self.peak_equity = total_equity_usd
            
        # Drawdown Check
        if self.peak_equity > 0:
            dd = (self.peak_equity - total_equity_usd) / self.peak_equity
            if dd > self.max_drawdown_pct:
                logger.critical(f"🚨 GLOBAL CIRCUIT BREAKER TRIPPED! Drawdown: {dd*100:.2f}% > {self.max_drawdown_pct*100:.2f}%")
                self.circuit_breaker_active = True
                self.liquidate_all_signal = True
            else:
                self.liquidate_all_signal = False
                
    def check_volatility_shock(self, asset_returns: Dict[str, float]) -> bool:
        """
        Check if assets are all crashing together (Correlation = 1).
        """
        if not asset_returns: return False
        
        # If all assets > 5 are down > 1% simultaneously -> Panic
        down_count = 0
        total_assets = len(asset_returns)
        
        for ret in asset_returns.values():
            if ret < -0.01: # 1% drop in candle
                down_count += 1
                
        if total_assets >= 3 and down_count == total_assets:
            logger.warning("🚨 SYSTEMIC SHOCK DETECTED: All assets crashing together.")
            return True
            
        return False
        
    def can_trade(self) -> bool:
        return not self.circuit_breaker_active
