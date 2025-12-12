from typing import Dict, List

import numpy as np
import pandas as pd


class MicrostructureFeatures:
    def __init__(self):
        self.prev_best_bid = None
        self.prev_best_ask = None
        self.prev_bid_vol = None
        self.prev_ask_vol = None

    def compute_features(self, snapshot: Dict) -> Dict[str, float]:
        """
        Compute alpha signals from L2 snapshot.
        """
        ob = snapshot["order_book"]
        if not ob["bids"] or not ob["asks"]:
            return {}

        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        bid_vol = ob["bids"][0][1]
        ask_vol = ob["asks"][0][1]
        mid_price = (best_bid + best_ask) / 2

        # 1. Microprice (Volume Weighted Mid Price)
        total_vol = bid_vol + ask_vol
        microprice = (
            (best_bid * ask_vol + best_ask * bid_vol) / total_vol
            if total_vol > 0
            else mid_price
        )

        # 2. Spread
        spread = best_ask - best_bid

        # 3. Order Flow Imbalance (OFI) - Simplified 1-level
        ofi = 0.0
        if self.prev_best_bid is not None:
            # Bid Side
            if best_bid > self.prev_best_bid:
                ofi += bid_vol
            elif best_bid == self.prev_best_bid:
                ofi += bid_vol - self.prev_bid_vol
            else:
                ofi -= self.prev_bid_vol

            # Ask Side
            if best_ask > self.prev_best_ask:
                ofi += self.prev_ask_vol
            elif best_ask == self.prev_best_ask:
                ofi -= ask_vol - self.prev_ask_vol
            else:
                ofi -= ask_vol

        # Update state
        self.prev_best_bid = best_bid
        self.prev_best_ask = best_ask
        self.prev_bid_vol = bid_vol
        self.prev_ask_vol = ask_vol

        # 4. Trade Features (CVD, VPIN proxy)
        trades = snapshot["recent_trades"]
        cvd_10s = 0.0
        volume_buy = 0.0
        volume_sell = 0.0

        now = snapshot["timestamp"]
        for t in trades:
            # Look back 10 seconds (10000 ms)
            if now - t["timestamp"] <= 10000:
                # is_buyer_maker = True -> Seller is Taker -> Sell Trade
                if t["is_buyer_maker"]:
                    volume_sell += t["quantity"]
                    cvd_10s -= t["quantity"]
                else:
                    volume_buy += t["quantity"]
                    cvd_10s += t["quantity"]

        vpin = (
            abs(volume_buy - volume_sell) / (volume_buy + volume_sell)
            if (volume_buy + volume_sell) > 0
            else 0
        )

        return {
            "microprice": microprice,
            "spread": spread,
            "ofi": ofi,
            "cvd_10s": cvd_10s,
            "vpin": vpin,
            "mid_price": mid_price,
        }
