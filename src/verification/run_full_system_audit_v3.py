import logging
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import replace

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("audit_v3")

from src.execution.backtest import Backtester, BacktestConfig
from src.risk_engine.risk_module import RiskEngine
from src.execution.live_signal_engine import LiveSignalEngine

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODEL_PATH = MODELS_DIR / "multifactor_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

def audit_pipeline_integrity():
    logger.info("CORE 1: Pipeline Integrity Check")
    
    # 1. Check Files
    required_files = [
        FEATURES_FILE, 
        MODEL_PATH, 
        SCALER_PATH,
        MODELS_DIR / "rl_agent_rf.pkl", # RL Agent
        MODELS_DIR / "multi_horizon_v1" / "metadata.json" # Multi-Horizon
    ]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        logger.error(f"❌ Missing Critical Files: {missing}")
        return False
        
    # 2. Schema Check
    try:
        df = pd.read_parquet(FEATURES_FILE)
        logger.info(f"✅ Features Loaded: Shape {df.shape}")

        
        # Check specific new Alpha columns
        required_cols = ["btc_alpha_trade_imbalance", "btc_alpha_vwap_reversal", "sentiment_divergence"]
        matching = [c for c in required_cols if c in df.columns]
        if len(matching) < len(required_cols):
             logger.warning(f"⚠️ Features missing some new alphas. Found: {matching}")
        else:
             logger.info("✅ New Alpha Sources Detected.")
             
    except Exception as e:
        logger.error(f"❌ Feature Load Failed: {e}")
        return False
        
    return True

def audit_strategy_performance():
    logger.info("CORE 2: Strategy Performance Audit (Backtest)")
    
    try:
        # Initialize Backtester
        config = BacktestConfig(
            taker_fee=0.0004, 
            maker_fee=0.0002,
            use_regime_filter=True
        )
        
        backtester = Backtester(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            features_path=FEATURES_FILE,
            config=config,
            initial_capital=10000.0
        )
        
        backtester.load_artifacts()
        backtester.prepare_data()
        backtester.run_backtest()
        metrics = backtester.calculate_metrics()
        
        logger.info("-------- Base Case Results --------")
        logger.info(json.dumps(metrics, indent=4))
        
        if metrics.get("profit_factor", 0) < 0.95:
             logger.warning("⚠️ Base Case Profit Factor < 0.95. Strategy is struggling.")
             return "WARN"
             
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Backtest Failed: {e}")
        import traceback
        traceback.print_exc()
        return "FAIL"

def audit_stress_tests():
    logger.info("CORE 3: Stress Testing (2x Fees + Slippage)")
    
    try:
        # 1. High Slippage + Fees
        config_stress = BacktestConfig(
            taker_fee=0.0008, # 2x Fees
            maker_fee=0.0004,
            slippage=0.001, # Doubled slippage
            use_regime_filter=True
        )
        
        backtester = Backtester(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            features_path=FEATURES_FILE,
            config=config_stress,
            initial_capital=10000.0
        )
        
        backtester.load_artifacts()
        backtester.prepare_data() # Re-runs predictions
        backtester.run_backtest()
        metrics = backtester.calculate_metrics()
        
        logger.info("-------- Stress Scenario Results --------")
        logger.info(json.dumps(metrics, indent=4))
        
        if metrics.get("max_drawdown", 0) < -0.25:
             logger.critical("❌ Stress Test Failed: Drawdown > 25%")
             return "FAIL"
             
        return metrics

    except Exception as e:
        logger.error(f"❌ Stress Test Failed: {e}")
        return "FAIL"

def audit_rl_ensemble():
    logger.info("CORE 3.5: RL Ensemble Verification")
    try:
        config = BacktestConfig(
            taker_fee=0.0004,
            use_regime_filter=True,
            use_rl_ensemble=True # Enable RL
        )
        
        backtester = Backtester(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            features_path=FEATURES_FILE,
            config=config,
            initial_capital=10000.0
        )
        
        backtester.load_artifacts()
        backtester.prepare_data()
        backtester.run_backtest()
        metrics = backtester.calculate_metrics()
        
        logger.info("-------- RL Ensemble Results --------")
        logger.info(json.dumps(metrics, indent=4))
        
        if metrics.get("profit_factor", 0) < 1.0:
             logger.warning("⚠️ RL Ensemble Profit Factor < 1.0.")
             
        return metrics
    except Exception as e:
        logger.error(f"❌ RL Ensemble Failed: {e}")
        return "FAIL"

def audit_risk_engine():
    logger.info("CORE 4: Risk Engine Verification")
    
    re = RiskEngine(account_size=10000)
    
    # Test 1: Volatility Sizing
    # Test 1: Volatility Sizing
    # Use high win rate to uncap Kelly. Use 20% vol to drop below Max Size Cap (10%).
    # 10% Vol hits the cap exactly (Risk/Vol = 1% / 10% = 10% Size). 20% should be 5%.
    size_low_vol = re.calculate_position_size(0.9, 50000, 0.01) # 1% vol
    size_high_vol = re.calculate_position_size(0.9, 50000, 0.20) # 20% vol
    
    logger.info(f"Position Size (1% Vol): {size_low_vol:.4f} BTC")
    logger.info(f"Position Size (20% Vol): {size_high_vol:.4f} BTC")
    
    if size_high_vol >= size_low_vol:
        logger.error("❌ Risk Engine Failed: Did not reduce size for high volatility.")
        return False
        
    # Test 2: Drawdown Circuit Breaker
    re.capital = 8000 # 20% Drawdown
    if not re.check_drawdown_limit():
        logger.info("✅ Circuit Breaker Triggered correctly at 20% DD.")
    else:
        logger.error("❌ Circuit Breaker Failed to trigger.")
        return False
        
    return True

def run_audit():
    results = {}
    
    # 1. Pipeline
    results["Pipeline"] = audit_pipeline_integrity()
    
    # 2. Base Strategy
    results["Base_Strategy"] = audit_strategy_performance()
    
    # 3. Stress Test
    results["Stress_Test"] = audit_stress_tests()
    
    # 3.5 RL Ensemble
    results["RL_Ensemble"] = audit_rl_ensemble()
    
    # 4. Risk Engine
    results["Risk_Engine"] = audit_risk_engine()
    
    print("\n\n======== AUDIT SUMMARY ========")
    for k, v in results.items():
        status = "✅ PASS"
        if v == False or v == "FAIL": status = "❌ FAIL"
        elif v == "WARN": status = "⚠️ WARN"
        elif isinstance(v, dict): status = f"✅ PASS (PF: {v.get('profit_factor', 0):.2f})"
        
        print(f"{k}: {status}")
        
if __name__ == "__main__":
    run_audit()
