import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.risk.portfolio_risk_v3 import PortfolioRiskEngine
from src.risk.scenarios import ScenarioSimulator
from src.risk.hedger import DynamicHedger
import pandas as pd
import numpy as np

def run_sim():
    print("🛡️ Starting Hedge Simulation...")
    
    # Setup
    engine = PortfolioRiskEngine()
    hedger = DynamicHedger(target_delta=0.0)
    
    # 1. Create Risky Portfolio
    print("\n[1] Initial State: Long 2 BTC (@ 50k)")
    positions = {"BTC": 2.0}
    prices = {"BTC": 50000.0}
    engine.update_portfolio(positions, prices, 150000) # 1.5x Leverage roughly
    
    # Mock History
    engine.calculate_risk_metrics(pd.DataFrame({"BTC": np.random.normal(0, 0.02, 100)}))
    metrics = engine.calculate_risk_metrics(pd.DataFrame({"BTC": np.random.normal(0, 0.02, 100)}))
    
    print(f"    Net Delta: ${metrics['net_delta']:,.2f}")
    print(f"    VaR (99%): ${metrics['var_usd']:,.2f}")
    
    # 2. Check Hedge
    print("\n[2] Hedger Analysis...")
    trades = hedger.propose_hedge(metrics, engine.positions)
    
    if trades:
        for t in trades:
            print(f"    🚨 PROPOSAL: {t['side']} {t['amount_usd']:,.2f} {t['instrument']} | Reason: {t['reason']}")
    else:
        print("    ✅ No hedges needed.")
        
    print("\n[3] Impact Simulation")
    # Apply Hedge (Simulated)
    hedge_amt = -trades[0]["amount_usd"] / 50000.0 # Short BTC
    positions["BTC_PERP_SHORT"] = hedge_amt # Add short
    prices["BTC_PERP_SHORT"] = 50000.0
    engine.update_portfolio(positions, prices, 150000)
    
    # Recalc
    new_metrics = engine.calculate_risk_metrics(pd.DataFrame({"BTC": np.random.normal(0, 0.02, 100)})) 
    print(f"    New Net Delta: ${new_metrics['net_delta']:,.2f}")
    print("    Risk Neutralized.")
    
    print("\n[4] Risk Guardian Integration Check...")
    from src.guardian.risk_guardian import RiskGuardian
    guardian = RiskGuardian()
    
    # Simulate CRASH scenario for Guardian
    # We already have 100k exposure. Let's see if Guardian triggers HARD_HEDGE
    # Guardian runs Scenario Check. 20% crash on 100k = -20k.
    # If Equity = 150k. -20k is 13% loss. Not > 20%.
    # Let's start with very high leverage for Guardian test
    print("    Simulating High Leverage (5x)...")
    # guardian.engine.update_portfolio({"BTC": 5.0}, ...) -> This updates internal state
    # But check_risk_state ALSO calls update_portfolio. So we just pass the raw dict.
    
    res = guardian.check_risk_state({"BTC": 5.0}, {"BTC": 50000.0}, 50000.0, pd.DataFrame({"BTC": np.random.normal(0, 0.02, 100)}))
    
    print(f"    Guardian State: {res['state']}")
    print(f"    Actions: {res['actions']}")
    
    if res['state'] == "HARD_HEDGE":
        print("    ✅ Guardian successfully triggered Hard Hedge on High Leverage.")


if __name__ == "__main__":
    run_sim()
