import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("arb_engine")

class ExchangeArbEngine:
    """
    v17 Cross-Exchange Edge Engine.
    Monitors Price Discrepancies between Exchanges.
    """
    def __init__(self, exchanges: List[str] = ["binance", "bybit", "okx"]):
        self.exchanges = exchanges
        
    def detect_arb(self, multi_exchange_depth: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Input: { 'binance': {'best_bid': X, 'best_ask': Y}, 'bybit': ... }
        Output: { 'opportunity': bool, 'spread': float, 'buy_ex': str, 'sell_ex': str }
        """
        best_bid_price = -1.0
        best_bid_ex = ""
        
        best_ask_price = float('inf')
        best_ask_ex = ""
        
        # Scan for Global Best Bid/Ask
        for ex, depth in multi_exchange_depth.items():
            if not depth: continue
            
            bb = depth.get("best_bid", 0.0)
            ba = depth.get("best_ask", float('inf'))
            
            if bb > best_bid_price:
                best_bid_price = bb
                best_bid_ex = ex
                
            if ba < best_ask_price:
                best_ask_price = ba
                best_ask_ex = ex
                
        # Check Cross_Exchange Spread (Arb)
        # Profit = Bid(High) - Ask(Low)
        if best_bid_price <= 0 or best_ask_price == float('inf'):
             return {"opportunity": False, "spread_pct": 0.0}
             
        spread = best_bid_price - best_ask_price
        spread_pct = spread / best_ask_price
        
        # Threshold: 0.03% (to cover fees)
        is_arb = spread_pct > 0.0003 
        
        if is_arb:
            logger.info(f"💎 ARB DETECTED: Buy {best_ask_ex} @ {best_ask_price} -> Sell {best_bid_ex} @ {best_bid_price} (Spread: {spread_pct*100:.3f}%)")
            
        return {
            "opportunity": is_arb,
            "spread_pct": spread_pct,
            "buy_exchange": best_ask_ex,
            "sell_exchange": best_bid_ex,
            "buy_price": best_ask_price,
            "sell_price": best_bid_price
        }
