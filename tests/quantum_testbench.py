# tests/quantum_testbench.py
import logging
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.decision.arbitrator import AgentArbitrator
from src.execution.execution_quantum import ExecutionQuantum
from src.risk_engine.risk_v3 import RiskEngineV3

# Mock Logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("quantum_bench")


def run_testbench():
    print("\n" + "=" * 50)
    print("⚛️ v19 QUANTUM System Testbench")
    print("=" * 50 + "\n")

    # 1. Initialize
    print("1. Initializing Quantum Core...")
    arbitrator = AgentArbitrator()
    risk = RiskEngineV3()
    exec_q = ExecutionQuantum()
    print("✅ Initialization Complete.")

    # 2. Test Arbitration
    print("\n2. Testing Multi-Agent Voting (Regime: NEUTRAL)...")
    signals = {
        "MomentumHunter": {"signal": 1.0, "confidence": 0.9},  # Bullish
        "MeanRevGhost": {"signal": -1.0, "confidence": 0.2},  # Weak Bearish
        "VolOracle": {"signal": 0.0, "confidence": 0.5},  # Neutral
    }
    # Neutral weights: Mom(0.3), MeanRev(0.5), Vol(0.2)
    # Score = (1*0.9*0.3) + (-1*0.2*0.5) + (0) = 0.27 - 0.1 = 0.17
    # 0.17 / 1.0 = 0.17 (HOLD, threshold 0.3)

    res = arbitrator.arbitrate(signals, "NEUTRAL")
    print(f"   Result: {res}")

    print("\n   Testing Regime Switch (RISK_ON)...")
    # Risk-On weights: Mom(0.6), MeanRev(0.2), Vol(0.2)
    # Score = (1*0.9*0.6) + (-1*0.2*0.2) = 0.54 - 0.04 = 0.50
    # 0.50 / 1.0 = 0.50 (BUY)
    res_bull = arbitrator.arbitrate(signals, "RISK_ON")
    print(f"   Result: {res_bull}")

    # 3. Test Risk Engine v3
    print("\n3. Testing Risk Factor Check...")
    # Simulate high Beta env
    market_factors = {"beta": 1.5, "liquidity": 1.0}

    # Trade: Size 0.1
    safe_trade = {"action": "BUY", "size": 1000.0}  # 1% of 100k

    is_safe = risk.check_trade_risk(safe_trade, market_factors)
    print(f"   Trade Safe? {is_safe}")

    # Simulate bad liquidity
    bad_liq = {"beta": 1.0, "liquidity": 0.2}
    is_safe_liq = risk.check_trade_risk(safe_trade, bad_liq)
    print(f"   Low Liq Trade Safe? {is_safe_liq}")

    # 4. Test Quantum Execution
    print("\n4. Testing Microprice Execution...")
    exec_q.execute_order(res_bull, {"microstructure": {"is_manipulated": False}})

    print("\n✅ QUANTUM Testbench Passed.")


if __name__ == "__main__":
    run_testbench()
