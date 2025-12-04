import asyncio
import json
import logging
import websockets
import time
from typing import Dict, List, Callable, Awaitable
from collections import deque
import numpy as np

logger = logging.getLogger("l2_stream")
logging.basicConfig(level=logging.INFO)

class L2Stream:
    def __init__(self, symbol: str = "btcusdt", callbacks: List[Callable[[Dict], Awaitable[None]]] = None):
        self.symbol = symbol.lower()
        self.ws_url = f"wss://fstream.binance.com/stream?streams={self.symbol}@depth20@100ms/{self.symbol}@aggTrade"
        self.callbacks = callbacks or []
        self.order_book = {"bids": [], "asks": []}
        self.trades = deque(maxlen=1000) # Keep last 1000 trades
        self.running = False

    async def connect(self):
        self.running = True
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info(f"Connected to {self.ws_url}")
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        await self.process_message(data)
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def process_message(self, data: Dict):
        stream = data.get("stream")
        payload = data.get("data")

        if not payload:
            return

        if "depth" in stream:
            self.update_order_book(payload)
        elif "aggTrade" in stream:
            self.update_trades(payload)

        # Notify callbacks with current state
        snapshot = self.get_snapshot()
        for callback in self.callbacks:
            await callback(snapshot)

    def update_order_book(self, payload: Dict):
        # For @depth20, payload is the snapshot itself (partial book)
        # Bids/Asks are lists of [price, quantity]
        self.order_book["bids"] = [[float(p), float(q)] for p, q in payload["b"]]
        self.order_book["asks"] = [[float(p), float(q)] for p, q in payload["a"]]
        self.order_book["timestamp"] = payload["E"]

    def update_trades(self, payload: Dict):
        trade = {
            "price": float(payload["p"]),
            "quantity": float(payload["q"]),
            "timestamp": payload["T"],
            "is_buyer_maker": payload["m"] # True if seller is taker (Buy), False if buyer is taker (Sell) -> Wait, m=True means Maker is Buyer -> Taker is Seller -> Sell Trade
        }
        self.trades.append(trade)

    def get_snapshot(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": time.time() * 1000,
            "order_book": self.order_book,
            "recent_trades": list(self.trades)
        }

    async def stop(self):
        self.running = False

# Example usage
async def print_snapshot(snapshot: Dict):
    ob = snapshot["order_book"]
    if ob["bids"] and ob["asks"]:
        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        logger.info(f"Bid: {best_bid} | Ask: {best_ask} | Trades: {len(snapshot['recent_trades'])}")

if __name__ == "__main__":
    stream = L2Stream(callbacks=[print_snapshot])
    try:
        asyncio.run(stream.connect())
    except KeyboardInterrupt:
        pass
