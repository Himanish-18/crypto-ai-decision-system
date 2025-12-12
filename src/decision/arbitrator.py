import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("arbitrator")


class AgentArbitrator:
    """
    v19 Multi-Agent Arbitration Layer.
    Uses Regime-Weighted Voting to combine agent signals.
    """

    def __init__(self):
        # Base Weights per regime
        self.regime_weights = {
            "RISK_ON": {"MomentumHunter": 0.6, "MeanRevGhost": 0.2, "VolOracle": 0.2},
            "RISK_OFF": {"MomentumHunter": 0.1, "MeanRevGhost": 0.3, "VolOracle": 0.6},
            "NEUTRAL": {"MomentumHunter": 0.3, "MeanRevGhost": 0.5, "VolOracle": 0.2},
            "LIQ_CRUNCH": {
                "MomentumHunter": 0.0,
                "MeanRevGhost": 0.0,
                "VolOracle": 1.0,
            },  # Defines strict defensive
        }
        
        # v40 Patch: Step 7 Reweight
        self.regime_weights["NEUTRAL"] = {
            "MomentumHunter": 0.25,
            "MeanRevGhost": 0.25,
            "ML_Ensemble": 0.25,
            "DOT_Signal": 0.25,
        }
        self.regime_weights["DOWN"] = {
            "MomentumHunter": 0.10,
            "MeanRevGhost": 0.10,
            "ML_Ensemble": 0.40,
            "DOT_Signal": 0.40,
        }
        self.regime_weights["UP"] = {
            "MomentumHunter": 0.10,
            "MeanRevGhost": 0.10,
            "ML_Ensemble": 0.40,
            "DOT_Signal": 0.40,
        }
        # Aliases
        self.regime_weights["BULL"] = self.regime_weights["UP"]
        self.regime_weights["BEAR"] = self.regime_weights["DOWN"]

    def arbitrate(
        self, agent_signals: Dict[str, Dict[str, Any]], regime: str
    ) -> Dict[str, Any]:
        """
        v46 Dynamic Arbitration.
        Uses Volatility-Adaptive Weights.
        """
        # Step 1: Extract Volatility (Normalized)
        metrics = agent_signals.get("MarketMetrics", {})
        vol_raw = metrics.get("volatility", 0.01) # Default 1%
        
        # Normalize Vol: 0.0 (Low) to 1.0 (High, >2%)
        vol_norm = min(max((vol_raw - 0.002) / 0.018, 0.0), 1.0) 
        
        # Step 2: Dynamic Weights (v46)
        # w_mom = 0.3 * vol_norm
        # w_mr = 0.3 * (1 - vol_norm)
        # w_ml = 0.2
        # w_dot = 0.2
        
        # Base Agents
        weights = {
            "MomentumHunter": 0.3 * (vol_norm),
            "MeanRevGhost": 0.3 * (1.0 - vol_norm),
            "ML_Ensemble": 0.2,
            "DOT_Signal": 0.2,
            "VolOracle": 0.0 # Placeholder
        }
        
        # Normalize weights to sum to 1.0 (ignoring empty agents)
        # Currently the formula sums to: 0.3*v + 0.3*(1-v) + 0.2 + 0.2 = 0.3 + 0.4 = 0.7.
        # It misses 0.3. User prompt: "w_mom=0.3*vol, w_mr=0.3*(1-vol), w_ml=0.2, w_dot=0.2".
        # Total = 0.7.
        # User implies these are the coefficients. I should Normalize them.
        # Or maybe User implies the remaining 0.3 goes somewhere else?
        # "Arbitrator output should be: decision = sign( Σ (signal_i * weight_i) )"
        # If sum(w) < 1, output is scaled down.
        # I will Normalize weights so they sum to 1.0 among active agents.
        
        active_weights = {k: v for k, v in weights.items() if k in agent_signals}
        total_w = sum(active_weights.values())
        if total_w > 0:
            norm_weights = {k: v / total_w for k, v in active_weights.items()}
        else:
            norm_weights = active_weights

        weighted_sum = 0.0
        weighted_conf = 0.0
        
        proposals = []

        for agent, w in norm_weights.items():
            res = agent_signals.get(agent, {})
            sig = res.get("signal", 0.0)
            conf = res.get("confidence", 0.0)
            
            # v46 Logic
            term = sig * w
            weighted_sum += term
            weighted_conf += conf * w
            
            proposals.append(f"{agent}({sig:.2f}|w={w:.2f})")
            
        # Step 3: Decision Logic
        # decision = sign(weighted_sum) -> mapped to Thresholds
        # User: decision > +0.25 -> BUY
        
        final_score = weighted_sum
        action = "HOLD"
        reason = f"DynamicArbitration(Vol={vol_raw:.4f}, Norm={vol_norm:.2f})"
        
        if final_score > 0.25:
            action = "BUY"
        elif final_score < -0.25:
            action = "SELL"
            
        # Size
        size = abs(weighted_conf)
        
        logger.info(
            f"⚖️ Arbitration: {proposals} -> Score: {final_score:.2f} -> {action}"
        )

        return {
            "action": action,
            "size": size, 
            "agent": "ARBITRATOR_v46",
            "reason": reason,
        }
