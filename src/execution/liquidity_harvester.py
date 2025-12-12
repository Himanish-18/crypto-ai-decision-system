
import logging
import time
from typing import Dict, Any, List, Optional
import random

from src.execution.fill_probability_model import FillProbabilityModel
from src.execution.queue_position_estimator import QueuePositionEstimator
from src.execution.slicer import OrderSlicer

logger = logging.getLogger("liquidity_harvester")

class LiquidityHarvester:
    """
    v43 Adaptive Harvesting Engine.
    Orchestrates: Slicing -> FillProb Check -> Maker/Taker Decision -> Queue Management.
    """
    def __init__(self, model_path=None, spread_recapture_target=0.15):
        self.fill_model = FillProbabilityModel(model_path)
        self.slicer = OrderSlicer()
        self.active_orders = {} # order_id -> QueueEstimator
        self.metrics = {
            "captured_spread": 0.0,
            "taker_fees_saved": 0.0,
            "adverse_selection_avoided": 0
        }
        self.spread_target = spread_recapture_target

    def execute_parent_order(self, symbol: str, side: str, qty: float, market_data: Dict[str, Any]):
        """
        Main entry point for execution.
        """
        logger.info(f"🌾 Harvesting Liquidity: {side} {qty} {symbol}")
        
        # 1. Slice
        clips = self.slicer.slice_order(qty)
        
        # 2. Sequential Execution Loop (Stub: In real HFT this is async/event-driven)
        # We simulate the decision process for the FIRST clip to demonstrate logic.
        curr_clip = clips[0]
        
        decision = self.decide_placement(side, curr_clip, market_data)
        logger.info(f"📋 First Clip Decision: {decision}")
        
        return decision

    def decide_placement(self, side: str, qty: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns: {action: 'POST' or 'TAKE', price: float, type: 'LIMIT'/'MARKET'}
        """
        # Feature Extraction
        # Spread, Skew, etc would come from market_data
        # Simulating data for decision
        
        best_bid = market_data.get("bid", 100.0)
        best_ask = market_data.get("ask", 100.5)
        mid_price = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid_price * 10000
        
        # 1. Fill Probability Check
        # State vector matching model training cols
        state = {
            "spread_bps": spread_bps,
            "depth_skew": 0.0, # Stub
            "volatility_1m": market_data.get("volatility", 0.01),
            "trade_flow_imbalance": 0.0,
            "dist_to_mid": 0.0 # At BBO
        }
        
        prob_fill = self.fill_model.predict_fill_prob(state)
        logger.info(f"📊 Fill Probability at BBO: {prob_fill:.4f}")
        
        # 2. Logic: Harvest vs Take
        # If Spread is wide and P(Fill) is decent -> POST PASSIVE
        # If Spread is tight (1 tick) or P(Fill) low -> TAKE AGGRESSIVE
        
        threshold = 0.40 # if > 40% chance to fill within horizon
        
        if prob_fill > threshold:
            price = best_bid if side == "BUY" else best_ask
            # Actually, to be passive maker:
            # Buy at Best Bid (Join Queue)
            # Sell at Best Ask (Join Queue)
            return {
                "action": "POST",
                "type": "LIMIT",
                "price": price,
                "qty": qty,
                "reason": f"High Fill Prob ({prob_fill:.2f})"
            }
        else:
            # Aggressive Take
            price = best_ask if side == "BUY" else best_bid
            return {
                "action": "TAKE",
                "type": "MARKET", # or Limit Crossing
                "price": price,
                "qty": qty,
                "reason": f"Low Fill Prob ({prob_fill:.2f})"
            }

    def on_market_update(self, order_id, update_type, data):
        """
        Callback for OrderBook updates to manage Queue Position.
        """
        if order_id in self.active_orders:
            q_est = self.active_orders[order_id]
            filled = q_est.update_market_event(update_type, data.get("qty", 0), data.get("side", ""))
            
            if filled:
                logger.info(f"✅ Order {order_id} estimated FILLED by Queue Model.")
                # Trigger logic to send next slice?
                del self.active_orders[order_id]
            else:
                # Check adverse selection risk
                # If price moves against us while we rest, cancel?
                pass
