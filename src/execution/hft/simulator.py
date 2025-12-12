import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("exec_sim")


class ExecutionSimulator:
    """
    v22 Execution Simulator.
    Backtests execution strategies against L2 snapshots.
    """

    def __init__(self):
        pass

    def run_simulation(self, snapshots: List[Dict], orders: List[Dict]) -> Dict:
        """
        snapshots: List of {"bids": [[price, qty], ...], "asks": [[price, qty], ...], "timestamp": ...}
        orders: List of {"side": "BUY", "price": float, "qty": float, "type": "LIMIT"}
        """
        fills = []
        slippage = []

        # Simple simulation:
        # For each order, scan forward in snapshots until filled.

        for order in orders:
            filled = False
            fill_price = 0.0
            fill_time = None

            target_price = order["price"]
            side = order["side"]

            for snap in snapshots:
                # Check crossing
                if side == "BUY":
                    best_ask = snap["asks"][0][0]  # Price
                    if order["type"] == "MARKET":
                        fill_price = best_ask * (1 + 0.0001)  # 1bp slippage assumption
                        filled = True
                    elif best_ask <= target_price:
                        fill_price = target_price  # Partial: Limit fill
                        filled = True
                else:  # SELL
                    best_bid = snap["bids"][0][0]
                    if order["type"] == "MARKET":
                        fill_price = best_bid * (1 - 0.0001)
                        filled = True
                    elif best_bid >= target_price:
                        fill_price = target_price
                        filled = True

                if filled:
                    fill_time = snap.get("timestamp")
                    fills.append(
                        {"order": order, "fill_price": fill_price, "time": fill_time}
                    )

                    # Calc Slippage vs Mid at arrival (assuming first snap is arrival)
                    mid_arrival = (
                        snapshots[0]["bids"][0][0] + snapshots[0]["asks"][0][0]
                    ) / 2
                    slip = (
                        (mid_arrival - fill_price)
                        if side == "SELL"
                        else (fill_price - mid_arrival)
                    )
                    slippage.append(slip / mid_arrival)
                    break

            if not filled:
                fills.append({"order": order, "fill_price": None, "status": "UNFILLED"})

        return {
            "fill_rate": (
                len([f for f in fills if f["fill_price"]]) / len(orders)
                if orders
                else 0
            ),
            "avg_slippage_bp": np.mean(slippage) * 10000 if slippage else 0.0,
            "fills": fills,
        }
