import sys
import os
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
import asyncio
import json

# Setup
PROJECT_ROOT = "/tmp/canary_v30_fix"
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.main as main_module

# Disable Execution
main_module.EXECUTION_MODE = "PAPER" 

def replay():
    print("🎬 Starting Deterministic Replay (Full Mocks v3)...")
    
    # Setup Loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Load Data
    data_path = "data/replay_samples/replay_2025-11-01.l3.parquet"
    df = pd.read_parquet(data_path)
    
    # Mock Everything that job() touches
    main_module.market_router = MagicMock()
    # Mock market_router_v2
    main_module.market_router_v2 = MagicMock()
    main_module.market_router_v2.scan_markets.return_value = "BTCUSDT"
    
    main_module.execution_quantum = MagicMock()
    
    main_module.self_healer = MagicMock()
    
    # Mock ob_manager
    main_module.ob_manager = MagicMock()
    main_module.ob_manager.get_latest_metrics.return_value = {"spread": 0.01}
    
    # Mock AI Components
    main_module.meta_brain = MagicMock()
    main_module.meta_brain.think.return_value = {"action": "HOLD", "reason": "Mock"}
    
    main_module.noise_guard = MagicMock()
    main_module.noise_guard.analyze_cleanliness.return_value = 0.0
    
    main_module.risk_engine_v3 = MagicMock()
    
    # Mock Liquidity AI
    main_module.liquidity_ai = MagicMock()
    main_module.liquidity_ai.analyze_intent.return_value = {"type": "PASSIVE"}
    
    main_module.smart_router = MagicMock()
    main_module.execution_v3 = MagicMock()
    main_module.shadow_agent = MagicMock()
    
    # Arbitrator
    try:
        from src.decision.arbitrator import AgentArbitrator
        main_module.arbitrator = AgentArbitrator()
        print("✅ Arbitrator Instantiated Real")
    except Exception as e:
        print(f"⚠️ Could not init Arbitrator: {e}")
        main_module.arbitrator = MagicMock()
        main_module.arbitrator.arbitrate.return_value = {"action": "BUY", "size": 1.0}
    
    # Ensure Globals exist (patched in main.py)
    if not hasattr(main_module, 'ml_model'):
        print("⚠️ ml_model missing in module!")
        
    decisions = []
    start_idx = 150
    steps = 5
    
    for i in range(steps):
        current_df = df.iloc[:start_idx + i].copy()
        
        f = loop.create_future()
        f.set_result(current_df)
        main_module.market_router.fetch_unified_candles.return_value = f
        
        try:
            main_module.job()
        except Exception as e:
            print(f"❌ Job Failed at step {i}: {e}")
            import traceback
            traceback.print_exc()
            
        decisions.append({
            "step": i,
            "pred": main_module.LAST_PRED_DIR,
            "score": main_module.LAST_PRED_SCORE
        })
        
    print(f"✅ Replay Complete. Decisions: {decisions}")
    
    bullish_count = sum(1 for d in decisions if d["pred"] in ["Bullish", "Buy", "Up"])
    if bullish_count > 0:
        print("✅ SUCCESS: Detected Bullish signal.")
    else:
        print(f"⚠️ WARNING: No Bullish signal. Got: {decisions}")
        
    if main_module.ml_model is not None:
        print("✅ ML Model valid.")

if __name__ == "__main__":
    replay()
