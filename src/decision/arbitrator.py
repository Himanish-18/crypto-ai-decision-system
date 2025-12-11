import logging
import numpy as np
from typing import Dict, List, Any

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
            "LIQ_CRUNCH": {"MomentumHunter": 0.0, "MeanRevGhost": 0.0, "VolOracle": 1.0}, # Defines strict defensive
        }

    def arbitrate(self, agent_signals: Dict[str, Dict[str, Any]], regime: str) -> Dict[str, Any]:
        """
        Combine signals using weighted voting.
        """
        weights = self.regime_weights.get(regime, self.regime_weights["NEUTRAL"])
        
        weighted_score = 0.0
        total_weight = 0.0
        
        proposals = []
        
        for agent_name, result in agent_signals.items():
            if agent_name not in weights: continue
            
            w = weights[agent_name]
            sig = result.get("signal", 0.0)
            conf = result.get("confidence", 0.0)
            
            # Vote = Signal * Confidence * Weight
            vote = sig * conf * w
            weighted_score += vote
            total_weight += w
            
            proposals.append(f"{agent_name}({sig:.2f})")
            
        if total_weight == 0:
            return {"action": "HOLD", "size": 0.0, "reason": "No weights"}
            
        final_score = weighted_score / total_weight
        
        # Decision Logic
        action = "HOLD"
        if final_score > 0.3: action = "BUY"
        elif final_score < -0.3: action = "SELL"
        
        logger.info(f"⚖️ Arbitration ({regime}): {proposals} -> Score: {final_score:.2f} -> {action}")
        
        return {
            "action": action,
            "size": abs(final_score), # Size proportional to conviction
            "agent": "ARBITRATOR_v19",
            "reason": f"Weighted Vote ({regime})"
        }
