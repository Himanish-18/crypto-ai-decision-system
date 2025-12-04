import asyncio
import logging
import json
from datetime import datetime
from typing import Dict

logger = logging.getLogger("micro_execution")

class MicroExecution:
    def __init__(self, log_file: str = "data/logs/l2_trades.jsonl"):
        self.log_file = log_file
        self.active_orders = {} # order_id -> {timestamp, price, side}

    async def execute(self, signal: Dict, snapshot: Dict):
        """
        Execute order based on signal and current L2 state.
        """
        direction = signal["direction"]
        confidence = signal["confidence"]
        
        if direction == 0 or confidence < 0.2:
            return

        # Risk Check: Spread Crossing
        ob = snapshot["order_book"]
        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        spread_pct = (best_ask - best_bid) / best_bid
        
        if spread_pct > 0.001: # 0.1% spread is too wide for micro-alpha
            logger.warning(f"Spread too wide ({spread_pct:.4f}), skipping trade.")
            return

        # Order Logic
        side = "BUY" if direction == 1 else "SELL"
        price = best_bid if side == "BUY" else best_ask # Passive/Limit entry
        
        # In a real system, we would call exchange API here
        # await exchange.create_limit_order(symbol, side, amount, price)
        
        # Simulation / Logging
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": snapshot["symbol"],
            "side": side,
            "price": price,
            "confidence": confidence,
            "features": {k: v for k, v in signal.items() if k != "direction"},
            "spread": spread_pct
        }
        
        await self.log_trade(trade_record)
        logger.info(f"Placed {side} order at {price} (Conf: {confidence:.2f})")

    async def log_trade(self, record: Dict):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    async def cancel_stale_orders(self):
        """
        Cancel orders older than 3 seconds.
        """
        now = datetime.now().timestamp()
        to_cancel = []
        for oid, order in self.active_orders.items():
            if now - order["timestamp"] > 3:
                to_cancel.append(oid)
        
        for oid in to_cancel:
            # await exchange.cancel_order(oid)
            del self.active_orders[oid]
            logger.info(f"Cancelled stale order {oid}")
