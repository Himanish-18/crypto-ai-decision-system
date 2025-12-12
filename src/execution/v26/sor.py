
from typing import Dict, List, Optional
import sys

class CrossExchangeSOR:
    """
    Smart Order Router to find best execution venue across exchanges.
    Objective: Min(Price + Fee + PredictedSlippage)
    """
    
    def __init__(self):
        # Mock Venues
        self.venues = ["Binance", "OKX", "Bybit"]
        self.fees = {
            "Binance": 0.0004, # Taker
            "OKX": 0.0005,
            "Bybit": 0.0006
        }
        
    def find_best_venue(self, side: str, qty: float, market_snapshot: Dict[str, Dict]) -> str:
        """
        snapshot: { "Binance": {"bid": 100, "ask": 101, "liq": 500}, ... }
        """
        best_venue = None
        best_cost = float('inf')
        
        for venue in self.venues:
            data = market_snapshot.get(venue)
            if not data: continue
            
            price = data['ask'] if side == 'BUY' else data['bid']
            liquidity = data.get('liq', 0)
            
            # Simple slippage penalty if liquidity < qty
            slippage_cost = 0.0
            if liquidity < qty:
                 slippage_cost = price * 0.001 * (qty / max(liquidity, 1.0))
            
            fee_cost = price * self.fees.get(venue, 0.0005)
            
            total_unit_cost = price + fee_cost + slippage_cost if side == 'BUY' else -(price - fee_cost - slippage_cost)
            
            # For BUY: Look for lowest total cost
            # For SELL: Look for highest net proceeds (lowest negative cost)
            
            score = total_unit_cost if side == 'BUY' else -total_unit_cost
            
            if score < best_cost:
                best_cost = score
                best_venue = venue
                
        return best_venue or "Binance"
