from typing import Dict, Any
import logging
from src.risk.portfolio_risk_v3 import PortfolioRiskEngine
from src.risk.hedger import DynamicHedger
from src.risk.scenarios import ScenarioSimulator

logger = logging.getLogger("risk_guardian")

class RiskGuardian:
    """
    v20 Risk Guardian.
    Monitors Portfolio Risk using RiskEngineV3.
    Triggers Hedge States:
    - NORMAL: No Action.
    - SOFT_HEDGE: Reduce Sizes.
    - HARD_HEDGE: Liquidate / Open Hedges.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.engine = PortfolioRiskEngine()
        self.hedger = DynamicHedger()
        self.simulator = ScenarioSimulator()
        
        self.max_var = self.config.get("max_var", 0.05) # 5% Daily VaR Limit
        self.max_leverage = self.config.get("max_leverage", 3.0)
        
    def check_risk_state(self, positions: Dict, prices: Dict, equity: float, history: Any) -> Dict[str, Any]:
        self.engine.update_portfolio(positions, prices, equity)
        metrics = self.engine.calculate_risk_metrics(history)
        
        state = "NORMAL"
        actions = []
        
        # 1. Check Hard Limits
        if metrics["var_99"] > self.max_var * 1.5:
            state = "HARD_HEDGE"
            actions.append("EMERGENCY_REDUCE: VaR Critical")
            # Propose Hedges
            hedges = self.hedger.propose_hedge(metrics, positions)
            actions.extend([f"EXECUTE: {h['side']} {h['instrument']}" for h in hedges])
            
        # 2. Check Soft Limits
        elif metrics["var_99"] > self.max_var:
            state = "SOFT_HEDGE"
            actions.append("STOP_OPENING: VaR Elevated")
            
        # 3. Check Scenarios
        scenarios = self.simulator.run_all(self.engine.positions)
        if scenarios.get("crypto_crash_20", 0) < -(equity * 0.20):
             state = "HARD_HEDGE" if state != "HARD_HEDGE" else state
             actions.append("SCENARIO_FAIL: Crash 20% wipes > 20% Equity")
             
        return {
            "state": state,
            "metrics": metrics,
            "actions": actions,
            "scenarios": scenarios
        }
