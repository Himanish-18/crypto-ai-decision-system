import logging
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger("hft_ob")


class HftOrderBook:
    """
    High-Frequency Order Book (L2) Reconstruction & Analytics.
    Maintains local state of Bids/Asks.
    Calculates: Micro-Imbalance, Weighted Mid, Book Pressure.
    """

    def __init__(self, symbol="BTC/USDT", depth_levels=20):
        self.symbol = symbol
        self.depth_levels = depth_levels

        # L2 State: {price: quantity}
        self.bids = defaultdict(float)
        self.asks = defaultdict(float)

        self.best_bid = 0.0
        self.best_ask = 0.0
        self.mid_price = 0.0

        # Metrics
        self.imbalance = 0.0
        self.spread = 0.0

    def update_snapshot(self, bids: list, asks: list):
        """
        Full overwrite of book state (e.g. from REST snapshot).
        bids/asks: list of [price, qty]
        """
        self.bids.clear()
        self.asks.clear()

        for p, q in bids:
            self.bids[p] = q

        for p, q in asks:
            self.asks[p] = q

        self._recalc_tops()

    def _recalc_tops(self):
        if not self.bids or not self.asks:
            return

        # Sort keys
        sorted_bids = sorted(self.bids.keys(), reverse=True)
        sorted_asks = sorted(self.asks.keys())

        self.best_bid = sorted_bids[0]
        self.best_ask = sorted_asks[0]
        self.spread = self.best_ask - self.best_bid
        self.mid_price = (self.best_ask + self.best_bid) / 2.0

    def get_imbalance(self, depth=5):
        """
        Calculates Order Book Imbalance at top N levels.
        Imbalance = (BidVol - AskVol) / (BidVol + AskVol)
        Range: [-1.0, 1.0]. Positive = Buying Pressure.
        """
        # Get top N levels
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[
            :depth
        ]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]

        bid_vol = sum([q for p, q in sorted_bids])
        ask_vol = sum([q for p, q in sorted_asks])

        if (bid_vol + ask_vol) == 0:
            return 0.0

        self.imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        return self.imbalance

    def get_weighted_mid_price(self, depth=5):
        """
        Volume-Weighted Mid Price (VWMP).
        Better reflection of "True Price" than simple mid.
        """
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[
            :depth
        ]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:depth]

        bid_vol = sum([q for p, q in sorted_bids])
        ask_vol = sum([q for p, q in sorted_asks])

        if not sorted_bids or not sorted_asks:
            return self.mid_price

        # Formula: (BestAsk * BidVol + BestBid * AskVol) / (BidVol + AskVol)
        # Standard Microstructure formula
        vwmp = (self.best_ask * bid_vol + self.best_bid * ask_vol) / (bid_vol + ask_vol)
        return vwmp

    def get_stats(self):
        return {
            "symbol": self.symbol,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "imbalance_5": self.get_imbalance(5),
            "vwmp": self.get_weighted_mid_price(5),
        }
