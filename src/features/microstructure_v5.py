import numpy as np
import pandas as pd


class MicrostructureV5:
    """
    v24 Institutional Microstructure Features.
    """

    @staticmethod
    def order_flow_imbalance(
        bids: pd.DataFrame, asks: pd.DataFrame, depth: int = 5
    ) -> float:
        """
        Calculates imbalance of top-N levels.
        OFI = (BidVol - AskVol) / (BidVol + AskVol)
        """
        # Stub: assuming df has bid_v_1..5, ask_v_1..5
        # Calculation logic
        total_bid = bids.iloc[:, :depth].sum(axis=1)
        total_ask = asks.iloc[:, :depth].sum(axis=1)

        return (total_bid - total_ask) / (total_bid + total_ask + 1e-9)

    @staticmethod
    def effective_spread_curvature(l1_spread: float, l5_spread: float) -> float:
        """
        Measures how fast liquidity falls off.
        """
        return l5_spread / (l1_spread + 1e-9)
