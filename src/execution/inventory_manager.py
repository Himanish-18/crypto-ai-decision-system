import logging
import numpy as np
from typing import Dict, Optional

# Setup Logging
logger = logging.getLogger("inventory_manager")

class InventoryManager:
    """
    Manages target inventory and position sizing based on PnL Skew and Risk.
    
    Rules:
    - Skew Rule: If Unrealized PnL < 0, reduce max exposure.
    - Pyramid Rule: If Unrealized PnL > 0, allow scaling up (up to a limit).
    """
    def __init__(self, max_capital: float = 10000.0, base_unit_size: float = 0.1):
        self.max_capital = max_capital
        self.base_unit_size = base_unit_size # Fraction of capital per trade
        self.current_exposure = 0.0
        self.unrealized_pnl = 0.0
        
    def update_state(self, exposure: float, unrealized_pnl: float):
        """Update current portfolio state."""
        self.current_exposure = exposure
        self.unrealized_pnl = unrealized_pnl

    def calculate_target_size(self, signal_strength: float, regime_multiplier: float = 1.0) -> float:
        """
        Calculate the target trade size (in quote currency).
        
        Args:
            signal_strength: 0.0 to 1.0 (Model confidence).
            regime_multiplier: Multiplier based on market regime (e.g., 0.5 for Chop, 1.0 for Trend).
        """
        # 1. Base Sizing
        base_size = self.max_capital * self.base_unit_size
        
        # 2. Apply Signal Strength
        # Linearly scale: 0.5 prob -> 0 size, 1.0 prob -> full size
        # Assuming signal_strength is > 0.5 for a trade
        confidence_factor = max(0, (signal_strength - 0.5) * 2)
        size = base_size * confidence_factor
        
        # 3. Apply Regime Multiplier
        size *= regime_multiplier
        
        # 4. Apply Skew Rule (Inventory Control)
        # If losing, reduce size to prevent revenge trading or digging deeper
        if self.unrealized_pnl < 0:
            # Decay factor: e.g., -1% PnL -> 0.9x size, -5% PnL -> 0.5x size
            pnl_pct = self.unrealized_pnl / self.max_capital
            skew_factor = max(0.1, 1.0 + (pnl_pct * 5)) # Aggressive reduction
            logger.info(f"📉 Losing Position (PnL: {pnl_pct*100:.2f}%). Skew Factor: {skew_factor:.2f}")
            size *= skew_factor
            
        elif self.unrealized_pnl > 0:
            # Pyramid: Allow slightly larger size if winning
            pnl_pct = self.unrealized_pnl / self.max_capital
            skew_factor = min(1.5, 1.0 + (pnl_pct * 2))
            logger.info(f"📈 Winning Position (PnL: {pnl_pct*100:.2f}%). Skew Factor: {skew_factor:.2f}")
            size *= skew_factor
            
        # 5. Cap at Max Capital
        if self.current_exposure + size > self.max_capital:
            size = max(0, self.max_capital - self.current_exposure)
            logger.warning(f"⚠️ Capped at Max Capital. Reduced size to {size}")
            
        return size

# Mock for Verification
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = InventoryManager(max_capital=10000, base_unit_size=0.1) # 1k base trade
    
    # Scenario 1: Neutral, High Confidence
    manager.update_state(exposure=0, unrealized_pnl=0)
    size = manager.calculate_target_size(signal_strength=0.8)
    logger.info(f"Scenario 1 Size: {size} (Expected ~600)")
    
    # Scenario 2: Losing, High Confidence
    manager.update_state(exposure=2000, unrealized_pnl=-100) # -1% PnL
    size = manager.calculate_target_size(signal_strength=0.8)
    logger.info(f"Scenario 2 Size: {size} (Expected < 600)")
    
    # Scenario 3: Winning, High Confidence
    manager.update_state(exposure=2000, unrealized_pnl=200) # +2% PnL
    size = manager.calculate_target_size(signal_strength=0.8)
    logger.info(f"Scenario 3 Size: {size} (Expected > 600)")
