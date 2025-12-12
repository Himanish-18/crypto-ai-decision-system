import sys
import os
import logging
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verifier")

def create_mock_df(trend="flat", length=500):
    """Create synthetic market data"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq="1min")
    base_price = 50000.0
    
    if trend == "up":
        prices = base_price * (1 + np.linspace(0, 0.05, length)) # +5%
    elif trend == "down":
        prices = base_price * (1 - np.linspace(0, 0.05, length)) # -5%
    else:
        prices = base_price + np.random.normal(0, 2, length)
        
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices + 5,
        "low": prices - 5,
        "close": prices,
        "volume": np.random.uniform(10, 100, length)
    })
    return df

def verify():
    print("🔬 STARTING v40 SIGNAL DIAGNOSTIC (MOCK INJECTION)...")
    
    # Import main module
    import src.main as main_module
    
    # --- INJECT MOCKS ---
    main_module.logger = MagicMock() # Supress logs
    
    # 1. Market Routers
    main_module.market_router_v2 = MagicMock()
    main_module.market_router_v2.scan_markets.return_value = "BTC/USDT"
    
    main_module.market_router = MagicMock()
    # fetch_unified_candles is async
    future = asyncio.Future()
    future.set_result(None) # Placeholder, will update per test
    main_module.market_router.fetch_unified_candles.return_value = future
    
    # 2. OB Manager
    main_module.ob_manager = MagicMock()
    main_module.ob_manager.get_latest_metrics.return_value = {"spread": 0.01}
    
    # 3. Noise Guard
    main_module.noise_guard = MagicMock()
    main_module.noise_guard.analyze_cleanliness.return_value = 0.0 # Clean
    
    # 4. Meta Brain
    main_module.meta_brain = MagicMock()
    main_module.meta_brain.think.return_value = {"action": "HOLD", "reason": "Test"}
    
    # 5. Arbitrator (Real Logic or Mock? Let's Mock to isolate Job Logic Wiring)
    # Actually, we want to test if signals are generated correctly. 
    # Job calls arbitrator. If we mock it, we don't see the result?
    # Job logs "Raw Signals" then calls arbi.
    # We want to see if `raw_signals` passed to arbi are correct.
    # So we Mock arbi and verify call args!
    main_module.arbitrator = MagicMock()
    main_module.arbitrator.arbitrate.return_value = {"action": "BUY", "size": 0.8} # Default
    
    # 6. Risk / Exec
    main_module.risk_engine_v3 = MagicMock()
    main_module.risk_engine_v3.check_trade_risk.return_value = True
    
    main_module.liquidity_ai = MagicMock()
    main_module.liquidity_ai.analyze_intent.return_value = {"type": "AGGRESSIVE"}
    
    main_module.execution_quantum = MagicMock()
    main_module.shadow_agent = None
    
    # 7. ML & DOT
    main_module.ml_model = MagicMock()
    # predict_proba returns [prob_down, prob_up] usually? Or [prob_0, prob_1]?
    # Assuming classes [0, 1].
    main_module.ml_model.predict_proba.return_value = np.array([[0.2, 0.8]]) # Bullish Default
    
    main_module.dot_model = MagicMock()
    main_module.dot_model.return_value = (torch.tensor([0.8]), None) # Bullish Default
    
    # 8. Globals
    main_module.TRAINING_FEATURES = ["btc_close", "btc_volume"] # Minimal acceptable
    
    # --- TEST LOOPS ---
    
    # Test 1: Strong Uptrend
    print("\n--- TEST 1: STRONG UPTREND ---")
    mock_df_up = create_mock_df("up")
    
    # Set Fetch Result
    fut = asyncio.Future()
    fut.set_result(mock_df_up)
    main_module.market_router.fetch_unified_candles.return_value = fut
    
    # Trigger Job
    try:
        main_module.job()
    except Exception as e:
        print(f"Job Failed: {e}")
        import traceback; traceback.print_exc()
        
    # Analyze
    # 1. Did it declare Market UP?
    # Logic: if ret1 > threshold. +5% over 500 bars is roughly 0.01% per bar?
    # 5% / 500 = 0.0001 per bar. 
    # Threshold is 0.0005. 
    # 0.0001 < 0.0005. So it will be FLAT!
    # I need Stronger trend for test.
    # Let's make it 50% over 500. = 0.1% per bar.
    # Update mock df generator locally or just manually.
    
    # Wait, 5% is strong trend. 0.0001 per minute is 0.01% per minute. 
    # 0.0005 is 0.05%.
    # 5% in 8 hours (500 mins) is realistic.
    # But checking per-bar return requires very sharp moves.
    # The threshold 0.0005 is quite high (5 basis points per minute).
    # BTC ATR(1) is usually 20-50 USD. Price 90k. 50/90000 = 0.00055.
    # So 0.0005 is reasonable.
    # My mock linear trend is too smooth.
    # I will inject a sharp move at the end.
    
    mock_df_up.iloc[-1, mock_df_up.columns.get_loc("close")] = mock_df_up.iloc[-2]["close"] * 1.001 # +0.1%
    fut = asyncio.Future()
    fut.set_result(mock_df_up)
    main_module.market_router.fetch_unified_candles.return_value = fut
    
    main_module.job()
    
    # Check Arbitrator Call
    call_args = main_module.arbitrator.arbitrate.call_args
    if call_args:
        signals, regime = call_args[0]
        print(f"Captured Signals: {signals}")
        print(f"Captured Regime: {regime}")
        
        if signals["MarketState"] == "UP":
            print("✅ Market State UP detected")
        else:
            print(f"❌ Market State Failed: {signals['MarketState']}")
            
        # Verify ML signal (Mocked to 0.8) -> Should be 1.0
        if signals["ML_Ensemble"]["signal"] == 1.0:
            print("✅ ML Signal Correct")
        else:
            print(f"❌ ML Signal Failed: {signals['ML_Ensemble']}")
            
    else:
        print("❌ Arbitrator not called!")

    print("\n✅ DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    verify()
