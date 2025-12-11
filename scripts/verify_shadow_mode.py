import sys
import logging
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock

# Set up logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_shadow")

def run_verification():
    logger.info("🕵️‍♂️ Starting Shadow Mode Verification...")
    
    # 1. Mock Dependencies & Globals
    import src.main as main_module
    
    # Mock Market Router
    mock_router = MagicMock()
    mock_router.fetch_unified_candles = AsyncMock(return_value=main_module.pd.DataFrame({
        "close": [90000.0 + i for i in range(100)],
        "volume": [100.0] * 100,
        "high": [90100.0] * 100,
        "low": [89900.0] * 100,
        "open": [90000.0] * 100
    }))
    main_module.market_router = mock_router
    
    # Mock Market Router V2
    mock_router_v2 = MagicMock()
    mock_router_v2.scan_markets = MagicMock(return_value="BTC/USDT")
    main_module.market_router_v2 = mock_router_v2
    
    # Mock OB Manager
    mock_ob = MagicMock()
    mock_ob.get_latest_metrics = MagicMock(return_value={"spread": 1.0, "imbalance": 0.5})
    main_module.ob_manager = mock_ob
    
    # Mock Noise Guard
    mock_noise = MagicMock()
    mock_noise.analyze_cleanliness = MagicMock(return_value=0.1) # Clean market
    main_module.noise_guard = mock_noise
    
    # Mock Self Healer
    main_module.self_healer = MagicMock()
    
    # Mock Liquidity AI
    mock_liq = MagicMock()
    mock_liq.analyze_intent = MagicMock(return_value={"type": "AGGRESSIVE"})
    main_module.liquidity_ai = mock_liq
    
    # Initialize Real Components (except those we want to test)
    # We want to test main_module.job()'s interaction with global objects
    # Instantiate globals that job() uses if they are None
    # Assuming main.py was imported but __main__ block not run
    
    from src.decision.meta_brain_v21 import MetaBrainV21
    from src.decision.arbitrator import AgentArbitrator
    from src.risk_engine.risk_v3 import RiskEngineV3
    from src.execution.execution_quantum import ExecutionQuantum
    from src.rl.ppo_portfolio import PPOPortfolioAgent
    
    main_module.meta_brain = MetaBrainV21()
    main_module.arbitrator = AgentArbitrator()
    main_module.risk_engine_v3 = RiskEngineV3()
    main_module.execution_quantum = ExecutionQuantum()
    
    # Force Enable Shadow Agent
    main_module.shadow_agent = PPOPortfolioAgent(state_dim=10, action_dim=4)
    
    # 2. Run Job
    logger.info("🚀 Running job()...")
    try:
        main_module.job()
        logger.info("✅ Job finished without error.")
    except Exception as e:
        logger.error(f"❌ Job Failed: {e}", exc_info=True)
        sys.exit(1)
        
    # 3. Check Logs (Manual Visual Confirmation via stdout capture by the tool)
    print("\nDid you see 'SHADOW PORTFOLIO' and 'Mode: ...' in the output above?")
    
if __name__ == "__main__":
    run_verification()
