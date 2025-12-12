import random  # Stub logic for prototype
from typing import Any, Dict

from src.agents.polymorph.base import PolymorphicAgent


class MomentumHunter(PolymorphicAgent):
    """
    Persona: Momentum Hunter.
    Strategy: Trades breakouts & vol bursts.
    """

    def __init__(self):
        super().__init__("MomentumHunter")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # Logic: If price > X% change & Volatility High -> Signal
        # Using stub logic for v16 scaffold
        candles = market_data.get("candles")
        if candles is None or candles.empty:
            return {"signal": 0, "confidence": 0, "risk_budget": 0, "latency_cost": 0}

        last_close = candles["close"].iloc[-1]
        prev_close = candles["close"].iloc[-2]
        pct_change = (last_close - prev_close) / prev_close

        signal = 0.0
        confidence = 0.0

        if abs(pct_change) > 0.005:  # Big move
            signal = 1.0 if pct_change > 0 else -1.0
            confidence = 0.85

        return {
            "signal": signal,
            "confidence": confidence,
            "risk_budget": 0.15,  # 15% budget
            "latency_cost": 50,  # 50ms (Model Inference)
            "meta": {"strategy": "breakout"},
        }


class MeanReversalGhost(PolymorphicAgent):
    """
    Persona: Mean Reversal Ghost.
    Strategy: Counter-trend scalping on OB imbalances.
    """

    def __init__(self):
        super().__init__("MeanReversalGhost")

    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # Logic: If OBI Extreme -> Fade
        obi = market_data.get("obi", 0.0)

        signal = 0.0
        confidence = 0.0

        if obi > 0.8:  # Huge Buy Wall -> Price might revert down? Or push up?
            # Standard mean rev assumption: Oversold/Overbought
            # Let's say RSI equivalent.
            signal = -1.0  # Short into strength
            confidence = 0.70
        elif obi < -0.8:
            signal = 1.0
            confidence = 0.70

        return {
            "signal": signal,
            "confidence": confidence,
            "risk_budget": 0.05,  # Scalp, low risk
            "latency_cost": 10,  # 10ms (Fast)
            "meta": {"strategy": "mean_rev"},
        }
