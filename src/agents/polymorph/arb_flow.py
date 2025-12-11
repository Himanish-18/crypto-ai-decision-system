from src.agents.polymorph.base import PolymorphicAgent
from typing import Dict, Any

class ArbSentinel(PolymorphicAgent):
    """
    Persona: Arb Sentinel.
    Strategy: Spreads BTC/ETH (Stat Arb) or Spot/Perp (Basis).
    """
    def __init__(self):
        super().__init__("ArbSentinel")
        
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # Logic: Compare Funding Rate or Spread
        # For prototype: Check if BTC price > 1.001 * ETH price scale (Stub)
        
        # Real Logic: Spot - Perp Basis
        basis = 0.0005 # Stub basis (5bps)
        funding = 0.0001 # Stub funding (1bps)
        
        signal = 0.0
        confidence = 0.0
        
        if basis > 0.002: # 20bps basis -> Short Perp / Long Spot
            signal = -1.0 
            confidence = 0.90 # Arb is high confidence
        
        return {
            "signal": signal,
            "confidence": confidence,
            "risk_budget": 0.30, # High budget for arb (low risk)
            "latency_cost": 5, # 5ms (Very fast calculation)
            "meta": {"strategy": "basis_arb"}
        }

class FlowReaper(PolymorphicAgent):
    """
    Persona: Flow Reaper.
    Strategy: Whale Detection & CVD Divergence.
    """
    def __init__(self):
        super().__init__("FlowReaper")
        
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        # Logic: CVD Divergence
        # If Price Down but CVD Up -> Bullish Divergence (Absorption)
        
        price_trend = -0.01 # Down
        cvd_trend = 500.0 # Positive Buying
        
        signal = 0.0
        confidence = 0.0
        
        if price_trend < 0 and cvd_trend > 0:
            signal = 1.0 # Buy the absorption
            confidence = 0.75
            
        return {
            "signal": signal,
            "confidence": confidence,
            "risk_budget": 0.20,
            "latency_cost": 20,
            "meta": {"strategy": "whale_cvd"}
        }
