import logging
import os
import sys

# Fix Path to include src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.execution.impact_model import MarketImpactModel
from src.ml.deep_stacker import DeepMetaStacker
from src.risk.stress_grid import StressGrid
from src.risk.var_engine import VarEngine

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")


def run_checks():
    logger.info("🛠 STARTING v24 PATCH VERIFICATION...")

    # 1. Check Risk
    stress = StressGrid()
    res = stress.run_stress_test({"BTC": 1.0}, 100000)
    logger.info(
        f"Risk Stress Scenarios run: {len(res)}. Max Loss: {res['Projected_Loss_Pct'].min():.2%}"
    )
    assert not res.empty, "Stress Grid returned empty"

    var_eng = VarEngine()
    dummy_ret = pd.DataFrame(np.random.normal(0, 0.01, (1000, 1)), columns=["BTC"])
    weights = np.array([1.0])
    v_res = var_eng.calculate_var(dummy_ret, weights)
    logger.info(f"VaR Engine Output: {v_res}")
    assert v_res["VaR_99"] < 0, "VaR should be negative"

    # 2. Check Execution
    impact = MarketImpactModel()
    imp_bps = impact.estimate_impact_bps(10000, 1000000, 0.02)
    logger.info(f"Impact of $10k on $1M DailyVol: {imp_bps:.4f} bps")
    assert imp_bps > 0, "Impact should be positive"

    # 3. Check ML
    stacker = DeepMetaStacker(5, 10)
    msg = "DeepStacker instantiated successfully"
    logger.info(msg)

    logger.info("✅ ALL SYSTEMS GREEN. INSTITUTIONAL PATCH VERIFIED.")


if __name__ == "__main__":
    try:
        run_checks()
    except Exception as e:
        logger.error(f"❌ VERIFICATION FAILED: {e}")
        sys.exit(1)
