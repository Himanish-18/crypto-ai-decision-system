import logging
from typing import Any, Dict

from src.execution.hft.fill_probability import FillProbabilityModel

logger = logging.getLogger("smart_router")


class SmartOrderRouter:
    """
    v22 Smart Order Router.
    Uses FillProbabilityModel for adaptive Maker/Taker decisions.
    """

    def __init__(self, fee_maker: float = 0.0002, fee_taker: float = 0.0005):
        self.fee_maker = fee_maker
        self.fee_taker = fee_taker
        self.fill_model = FillProbabilityModel()
        self.max_slice_qty = 1.0  # arbitrary cap for demo

    def route_order(
        self,
        side: str,
        qty: float,
        market_data: Dict[str, Any],
        urgency: str = "NORMAL",
    ) -> Dict[str, Any]:
        """
        Returns execution instructions.
        """
        # Slicing Logic
        if qty > self.max_slice_qty and urgency != "HIGH":
            logger.info(f"🔪 Slicing Large Order {qty} -> {self.max_slice_qty}")
            qty = self.max_slice_qty  # Simple cap for this routable chunk

        ob_imb = market_data.get("ob_imbalance", 0.0)
        # normalize imb to [-1, 1] if needed, assuming input is already scaled

        # Predict Prob at best bid/ask (distance ~ 0)
        prob_maker = self.fill_model.predict(side, distance_pct=0.0, obi=ob_imb)

        # Volatility Penalty
        vol = market_data.get("volatility", 0.0)
        if vol > 0.005:
            prob_maker *= 0.8

        # Urgency Penalty
        if urgency == "HIGH":
            prob_maker -= 0.3

        threshold = 0.6

        if prob_maker > threshold:
            logger.info(f"⏳ SOR: Routing MAKER (Prob {prob_maker:.2f} > {threshold}).")
            return {"type": "LIMIT", "side": side, "qty": qty, "exec_inst": "POST_ONLY"}
        else:
            logger.info(f"⚡ SOR: Routing TAKER (Prob {prob_maker:.2f} < {threshold}).")
            return {"type": "MARKET", "side": side, "qty": qty}
