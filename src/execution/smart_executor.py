import logging
import time
import asyncio
from typing import Dict, Optional
from src.execution.binance_executor import BinanceExecutor

# Setup Logging
logger = logging.getLogger("smart_executor")

class SmartExecutor:
    """
    Smart Execution Engine for optimizing trade entry and exit.
    Features:
    - Passive Entry (Maker) with Queue Position Estimation.
    - Cancel + Replace logic.
    - Aggressive Fallback (Taker).
    """
    def __init__(self, executor: BinanceExecutor):
        self.executor = executor
        self.active_orders = {} # order_id -> order_details
        
        # Config
        self.max_wait_time = 10 # Seconds to wait for fill
        self.replace_threshold = 0.0005 # 5 bps price move triggers replace
        
    async def execute_order(self, symbol: str, side: str, amount: float, price: Optional[float] = None, style: str = "PASSIVE"):
        """
        Execute an order with the specified style.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT").
            side: "buy" or "sell".
            amount: Quantity to trade.
            price: Limit price (optional). If None, uses current best bid/ask.
            style: "PASSIVE" (Maker) or "AGGRESSIVE" (Taker).
        """
        logger.info(f"🚀 Executing {style} {side.upper()} {amount} {symbol}")
        
        if style == "AGGRESSIVE":
            return self.executor.place_order(symbol, side, amount, order_type="market")
            
        elif style == "PASSIVE":
            return await self._execute_passive(symbol, side, amount, price)
            
        else:
            raise ValueError(f"Unknown execution style: {style}")

    async def _execute_passive(self, symbol: str, side: str, amount: float, price: Optional[float]):
        """
        Handle passive execution loop: Place Limit -> Monitor -> Replace/Fill.
        """
        # 1. Determine Initial Price
        if price is None:
            ticker = self.executor.get_ticker(symbol)
            if side == "buy":
                price = ticker["bid"] # Join Best Bid
            else:
                price = ticker["ask"] # Join Best Ask
                
        # 2. Place Initial Limit Order
        order = self.executor.place_order(symbol, side, amount, order_type="limit", price=price)
        if not order:
            logger.error("Failed to place initial passive order.")
            return None
            
        order_id = order["id"]
        start_time = time.time()
        initial_volume_ahead = self._estimate_queue_position(symbol, side, price)
        
        logger.info(f"⏳ Passive Order {order_id} placed at {price}. Est. Queue: {initial_volume_ahead:.4f}")
        
        # 3. Monitor Loop
        while True:
            await asyncio.sleep(1) # Check every second
            
            # Check Status
            status = self.executor.get_order_status(symbol, order_id)
            if status == "filled":
                logger.info(f"✅ Order {order_id} FILLED!")
                return order
            elif status == "canceled":
                logger.warning(f"❌ Order {order_id} CANCELED externally.")
                return None
                
            # Check Timeout -> Aggressive Fallback
            if time.time() - start_time > self.max_wait_time:
                logger.info(f"⏰ Timeout ({self.max_wait_time}s). Switching to AGGRESSIVE.")
                self.executor.cancel_order(symbol, order_id)
                return self.executor.place_order(symbol, side, amount, order_type="market")
                
            # Check Price Move -> Cancel & Replace
            ticker = self.executor.get_ticker(symbol)
            current_best = ticker["bid"] if side == "buy" else ticker["ask"]
            
            dist = abs(current_best - price) / price
            if dist > self.replace_threshold:
                logger.info(f"🔄 Price moved {dist*100:.4f}%. Replacing order...")
                self.executor.cancel_order(symbol, order_id)
                # Recurse with new price
                return await self._execute_passive(symbol, side, amount, None)

    def _estimate_queue_position(self, symbol: str, side: str, price: float) -> float:
        """
        Estimate volume ahead in the queue.
        Simple heuristic: Volume at the price level in the order book.
        """
        # In a real system, we would snapshot the book right before placing.
        # Here we fetch the book again (slight race condition, but acceptable for alpha).
        try:
            book = self.executor.exchange.fetch_order_book(symbol, limit=5)
            if side == "buy":
                for p, q in book["bids"]:
                    if abs(p - price) < 1e-8:
                        return q
            else:
                for p, q in book["asks"]:
                    if abs(p - price) < 1e-8:
                        return q
        except Exception as e:
            logger.warning(f"Failed to estimate queue: {e}")
            
        return 0.0

# Mock for Verification
if __name__ == "__main__":
    # Mock Executor
    class MockExecutor:
        def __init__(self):
            self.exchange = self
        def get_ticker(self, symbol):
            return {"bid": 100.0, "ask": 100.1}
        def place_order(self, symbol, side, amount, order_type, price=None):
            return {"id": "123", "status": "open"}
        def get_order_status(self, symbol, order_id):
            # Simulate fill after 2 calls
            if not hasattr(self, "calls"): self.calls = 0
            self.calls += 1
            return "filled" if self.calls > 2 else "open"
        def cancel_order(self, symbol, order_id):
            pass
        def fetch_order_book(self, symbol, limit):
            return {"bids": [[100.0, 5.0]], "asks": [[100.1, 2.0]]}

    async def main():
        mock_exec = MockExecutor()
        smart_exec = SmartExecutor(mock_exec)
        await smart_exec.execute_order("BTC/USDT", "buy", 0.1, style="PASSIVE")

    asyncio.run(main())
