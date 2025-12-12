from typing import Any, Dict, List

import numpy as np
import pandas as pd


class MicrostructureFeatures:
    def __init__(self):
        pass

    def calculate_order_imbalance(
        self, order_book: Dict[str, Any], depth: int = 10
    ) -> float:
        """
        Calculate Order Imbalance Ratio.
        (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
        """
        bids = np.array(order_book["bids"])[:depth]
        asks = np.array(order_book["asks"])[:depth]

        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])

        if bid_vol + ask_vol == 0:
            return 0.0

        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def calculate_cvd(self, trades: pd.DataFrame) -> float:
        """
        Calculate Cumulative Volume Delta for the given trades.
        Returns the net delta (Buy Vol - Sell Vol).
        """
        if trades.empty:
            return 0.0

        # 'side' is usually 'buy' or 'sell'
        buy_vol = trades[trades["side"] == "buy"]["amount"].sum()
        sell_vol = trades[trades["side"] == "sell"]["amount"].sum()

        return buy_vol - sell_vol

    def calculate_vwap(self, trades: pd.DataFrame) -> float:
        """
        Calculate VWAP from trades.
        Sum(Price * Volume) / Sum(Volume)
        """
        if trades.empty:
            return 0.0

        total_vol = trades["amount"].sum()
        if total_vol == 0:
            return 0.0

        vwap = (trades["price"] * trades["amount"]).sum() / total_vol
        return vwap

    def calculate_features(
        self, order_book: Dict[str, Any], trades: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate all microstructure features.
        """
        imbalance_10 = self.calculate_order_imbalance(order_book, depth=10)
        imbalance_50 = self.calculate_order_imbalance(order_book, depth=50)

        cvd = self.calculate_cvd(trades)
        vwap = self.calculate_vwap(trades)

        current_price = trades.iloc[-1]["price"] if not trades.empty else 0
        vwap_deviation = (current_price - vwap) / vwap if vwap != 0 else 0

        return {
            "order_imbalance_10": imbalance_10,
            "order_imbalance_50": imbalance_50,
            "cvd_1h": cvd,  # Assuming trades passed are from last 1h
            "vwap_deviation": vwap_deviation,
        }
