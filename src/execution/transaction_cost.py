import logging
import numpy as np
from typing import Dict

# Setup Logging
logger = logging.getLogger("transaction_cost")

class TransactionCostModel:
    """
    Estimates transaction costs including Exchange Fees and Slippage.
    """
    def __init__(self, maker_fee: float = -0.0002, taker_fee: float = 0.0005):
        # Default: Maker Rebate 2bps, Taker Fee 5bps
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # Asset-specific slippage parameters (Linear Impact Model)
        # Slippage (bps) = Alpha * (OrderSize / DailyVolume)
        self.slippage_params = {
            "BTC/USDT": {"alpha": 0.1, "daily_vol": 10_000_000_000},
            "ETH/USDT": {"alpha": 0.15, "daily_vol": 5_000_000_000},
            "BNB/USDT": {"alpha": 0.3, "daily_vol": 1_000_000_000},
        }

    def estimate_cost(self, symbol: str, side: str, amount: float, price: float, style: str = "PASSIVE") -> float:
        """
        Estimate total cost of the trade in quote currency.
        
        Args:
            style: "PASSIVE" (Maker) or "AGGRESSIVE" (Taker).
        """
        notional = amount * price
        
        # 1. Exchange Fees
        if style == "PASSIVE":
            fee_rate = self.maker_fee
        else:
            fee_rate = self.taker_fee
            
        fee_cost = notional * fee_rate
        
        # 2. Slippage (Market Impact)
        # Only applies to Aggressive (Taker) orders significantly.
        # Passive orders might have "adverse selection" but we model that separately.
        # Here we model immediate execution cost.
        slippage_cost = 0.0
        if style == "AGGRESSIVE":
            params = self.slippage_params.get(symbol, {"alpha": 0.5, "daily_vol": 100_000_000})
            impact_bps = params["alpha"] * np.sqrt(notional / params["daily_vol"])
            slippage_cost = notional * impact_bps
            
        total_cost = fee_cost + slippage_cost
        
        logger.debug(f"💰 Cost Est for {symbol} ({style}): Fee=${fee_cost:.4f}, Slippage=${slippage_cost:.4f}, Total=${total_cost:.4f}")
        return total_cost

# Mock for Verification
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tcm = TransactionCostModel()
    
    # Test Passive (Rebate)
    cost_passive = tcm.estimate_cost("BTC/USDT", "buy", 1.0, 50000, style="PASSIVE")
    logger.info(f"Passive Cost (1 BTC): ${cost_passive:.4f} (Should be negative/rebate)")
    
    # Test Aggressive (Fee + Slippage)
    cost_aggressive = tcm.estimate_cost("BTC/USDT", "buy", 1.0, 50000, style="AGGRESSIVE")
    logger.info(f"Aggressive Cost (1 BTC): ${cost_aggressive:.4f}")
    
    # Test Large Aggressive (High Slippage)
    cost_whale = tcm.estimate_cost("BTC/USDT", "buy", 100.0, 50000, style="AGGRESSIVE")
    logger.info(f"Whale Cost (100 BTC): ${cost_whale:.4f}")
