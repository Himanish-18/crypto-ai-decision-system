import logging
import random
from typing import Dict, List, Any
from src.agents.polymorph.base import PolymorphicAgent
# Import Agents
from src.agents.polymorph.momentum_reversal import MomentumHunter, MeanReversalGhost
from src.agents.polymorph.arb_flow import ArbSentinel, FlowReaper
from src.agents.polymorph.vol_carry import VolatilityOracle, CarryArchitect

logger = logging.getLogger("meta_brain")

class MetaBrain:
    """
    v16 Master Router.
    Aggregates signals from Polymorphic Agents.
    Allocates capital via Dynamic Capital Flow Optimizer (DCFO).
    """
    def __init__(self):
        self.agents: List[PolymorphicAgent] = [
            MomentumHunter(), MeanReversalGhost(), ArbSentinel(),
            FlowReaper(), VolatilityOracle(), CarryArchitect()
        ]
        self.regime = "NORMAL" # TRENDING, RANGING, NOISE
        
    def think(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all agents and choose best action.
        """
        candidates = []
        
        # 1. Query all agents
        for agent in self.agents:
            res = agent.analyze(market_data)
            signal = res.get("signal", 0.0)
            conf = res.get("confidence", 0.0)
            
            # --- FILTER ---
            # Ignore low confidence
            if conf < 0.55:
                continue
                
            # Ignore Latency Risk (Stub: if cost > 100ms and HFT needed)
            # if res["latency_cost"] > 100 ...
            
            if abs(signal) > 0:
                candidates.append({
                    "name": agent.name,
                    "signal": signal,
                    "conf": conf,
                    "budget": res.get("risk_budget", 0.1)
                })
                
        if not candidates:
            return {"action": "HOLD", "reason": "No high confidence signals"}
            
        # 2. Meta-Selection Strategy (Winner Takes Most or Ensemble)
        # For v16, we pick Highest Confidence
        best = max(candidates, key=lambda x: x["conf"])
        
        # 3. Dynamic Capital Flow (DCFO)
        # Scale size by Confidence
        # Base Allocation 10% * Confidence Scaler
        size = best["budget"] * (best["conf"] / 0.8) 
        size = min(size, 0.4) # Max 40% per trade
        
        logger.info(f"🧠 Meta-Brain Selects: {best['name']} | Sig: {best['signal']} | Conf: {best['conf']:.2f}")
        
        return {
            "action": "BUY" if best["signal"] > 0 else "SELL",
            "size": size,
            "agent": best["name"],
            "reason": f"Selected {best['name']} (Conf {best['conf']:.2f})"
        }
