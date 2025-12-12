import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.risk_engine.risk_module import RiskEngine

logger = logging.getLogger("trading_decision")


class TradingDecision:
    def __init__(self, risk_engine: RiskEngine, log_dir: Path):
        self.risk_engine = risk_engine
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def make_decision(
        self,
        signal_output: Dict[str, Any],
        current_price: float,
        win_rate: float = 0.39,
        microstructure: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Combine signal, risk, and microstructure to make a trading decision.
        """
        timestamp = signal_output["timestamp"]
        prob = signal_output["prediction_prob"]
        signal = signal_output["signal"]
        context = signal_output["strategy_context"]

        decision = {
            "timestamp": str(timestamp),
            "action": "HOLD",
            "size": 0.0,
            "stops": {},
            "reason": "No Signal",
        }

        # 1. Check Filters
        is_allowed, reason = self.risk_engine.check_filters(context)
        if not is_allowed:
            decision["reason"] = f"Filtered: {reason}"
            self.log_decision(decision, signal_output)
            return decision

        # 2. Check Signal & Microstructure Confirmation
        if signal == 1:
            # Alpha Upgrade: Orderflow Confirmation
            confirmed = True
            if microstructure:
                # Require positive CVD for Buy
                if microstructure.get("cvd_1h", 0) < 0:
                    confirmed = False
                    decision["reason"] = "Signal Rejected: Negative CVD"

                # Require Order Imbalance favoring execution side
                obi = microstructure.get("obi", 0)
                if signal == 1 and obi < -0.3:  # Trying to buy into heavy sell pressure
                    # But wait, maybe contrarian?
                    # Goal: Reduce slippage/risk. If book is heavy Sell, price might drop.
                    # We skip trade.
                    confirmed = False
                    decision["reason"] = (
                        f"Signal Rejected: OBI ({obi:.2f}) indicates Sell Pressure"
                    )

            if confirmed:
                # 3. Calculate Size
                size = self.risk_engine.calculate_position_size(win_rate, current_price)

                if size > 0:
                    decision["action"] = "BUY"
                    decision["size"] = size
                    decision["stops"] = self.risk_engine.get_exit_params(current_price)
                    decision["reason"] = (
                        f"Signal Buy (Prob {prob:.2f}) + Orderflow Confirmed"
                    )
                else:
                    decision["reason"] = "Zero Size (Risk/Kelly)"

        self.log_decision(decision, signal_output)
        return decision

    def log_decision(self, decision: Dict, signal_output: Dict):
        """Log decision to JSON."""

        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        decision_log = {k: convert_numpy(v) for k, v in decision.items()}
        decision_log["timestamp"] = str(decision["timestamp"])

        signal_log = {k: convert_numpy(v) for k, v in signal_output.items()}
        signal_log["timestamp"] = str(signal_output["timestamp"])

        if "strategy_context" in signal_log:
            signal_log["strategy_context"] = {
                k: convert_numpy(v) for k, v in signal_log["strategy_context"].items()
            }

        log_entry = {"decision": decision_log, "signal": signal_log}

        log_file = self.log_dir / "trading_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
