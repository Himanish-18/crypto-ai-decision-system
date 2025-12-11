import asyncio
import logging
import time
from typing import Dict, List, Optional
from collections import deque
import numpy as np

logger = logging.getLogger("hft_orderbook")

class WebSocketOrderBook:
    """
    Maintains a local high-frequency order book (L2) by replaying WebSocket diffs.
    Architecture:
    1. Snapshot Init: Fetch REST snapshot.
    2. Buffer: Queue WS events that arrive during snapshot fetch.
    3. Replay: Apply buffered events.
    4. Real-time: Apply new events immediately.
    """
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.bids: Dict[float, float] = {} # Price -> Size
        self.asks: Dict[float, float] = {}
        
        self.last_update_id = 0
        self.is_synced = False
        self.lock = asyncio.Lock()
        
        # Metrics
        self.update_count = 0
        self.latency_stats = deque(maxlen=1000)
        
    def apply_diff(self, bids: List[List[str]], asks: List[List[str]], u_id: int, pu_id: int):
        """
        Apply standard [price, quantity] diff.
        Quantity 0 means delete level.
        """
        if u_id <= self.last_update_id:
            return # Stale packet
            
        # Basic gap check (implementation depends on exchange, e.g. Binance u_id vs pu_id)
        # For this prototype, we assume correct ordering from WS Manager
        
        for price_str, qty_str in bids:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                if price in self.bids: del self.bids[price]
            else:
                self.bids[price] = qty
                
        for price_str, qty_str in asks:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                if price in self.asks: del self.asks[price]
            else:
                self.asks[price] = qty
                
        self.last_update_id = u_id
        self.update_count += 1
        
    def update_snapshot(self, bids: List[List[float]], asks: List[List[float]]):
        """
        Updates the book from a partial depth snapshot (e.g. depth20).
        Replaces current dicts with new snapshot data.
        """
        self.bids = {float(p): float(q) for p, q in bids}
        self.asks = {float(p): float(q) for p, q in asks}
        self.update_count += 1
        
    def get_snapshot(self, limit: int = 10) -> Dict:
        """
        Returns sorted Bids/Asks tops.
        """
        # Sort Bids Desc
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:limit]
        # Sort Asks Asc
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:limit]
        
        return {
            "bids": sorted_bids,
            "asks": sorted_asks,
            "ts": time.time_ns()
        }
        
    def get_mid_price(self) -> float:
        if not self.bids or not self.asks: return 0.0
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return (best_bid + best_ask) / 2.0
        
    def get_imbalance(self, depth: int = 5) -> float:
        """
        Calculate Order Book Imbalance (VOI).
        (BidVol - AskVol) / (BidVol + AskVol)
        """
        snap = self.get_snapshot(limit=depth)
        bid_vol = sum([x[1] for x in snap['bids']])
        ask_vol = sum([x[1] for x in snap['asks']])
        total = bid_vol + ask_vol + 1e-9
        return (bid_vol - ask_vol) / total
