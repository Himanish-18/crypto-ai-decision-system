
import sys
import os
import logging
import pandas as pd
import numpy as np
import asyncio
import torch
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Mock dependencies to avoid real connections
sys.modules['src.data.market_router'] = MagicMock()
sys.modules['src.execution.binance_executor'] = MagicMock()
sys.modules['src.execution.live_signal_engine'] = MagicMock()
sys.modules['src.execution.smart_executor'] = MagicMock()
sys.modules['src.execution.trading_decision'] = MagicMock()
sys.modules['src.features.alpha_signals'] = MagicMock()
sys.modules['src.features.build_features'] = MagicMock()
# We need REAL build_features logic though to test if verify worked? 
# Actually main.py imports explicit functions from build_features.
# We should allow those imports if possible, or mock them to return KNOWN data to verify logic.

# Let's import main after mocking heavy external things
# But we want main to use the REAL calculate_features if we want to test that call.
# However, calculate_features relies on ta library.

# Better approach: Mock the Data Fetching in main.py, but let the rest run.
# We need to mock:
# src.data.market_router.MarketRouter
# src.market.router_v2.MarketRouterV2
# src.features.orderbook_features.OrderBookManager
# src.decision.meta_brain_titan (if it makes network calls)
# src.decision.meta_brain_v21

from src import main

# Mock Config
main.config = {}

# Mock Objects in main
main.market_router = MagicMock()
main.market_router_v2 = MagicMock()
main.ob_manager = MagicMock()
main.meta_brain = MagicMock()
main.arbitrator = MagicMock() # We want to test logic BEFORE arbitrator or verify arbitrator output
# Wait, main calls arbitrator.arbitrate(). The logic I changed is preparing signals passed TO arbitrator.

# I need to inspect the 'raw_signals' passed to arbitrator.
# I can attach a side_effect to arbitrator.arbitrate to print its input.

def capture_signals(signals, regime):
    print(f"CAPTURED SIGNALS: {signals}")
    return {"action": "BUY", "size": 1.0, "agent": "TEST"}

main.arbitrator.arbitrate.side_effect = capture_signals

# Mock MetaBrain think to return Safe
main.meta_brain.think.return_value = {"action": "PASS", "agent": "TITAN_Safe"}

# Mock Noise Guard
main.noise_guard = MagicMock()
main.noise_guard.analyze_cleanliness.return_value = 0.1 # Clean

# Mock Market Data Fetch
# We need to return a DF that triggers a Buy or Sell.
# RSI < 30 -> Buy.
# Let's construct a DF that results in RSI < 30 or Price < BB Low.

def get_market_data(*args, **kwargs):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1min')
    df = pd.DataFrame({
        'open': np.full(100, 50000.0),
        'high': np.full(100, 51000.0),
        'low': np.full(100, 49000.0),
        'close': np.linspace(60000, 30000, 100), # crashing price to trigger RSI oversold
        'volume': np.full(100, 100.0)
    }, index=dates)
    return df

async def mock_fetch(*args, **kwargs):
    return get_market_data()

main.market_router.fetch_unified_candles = MagicMock(side_effect=mock_fetch)

# Mock Market Router V2 Scan
main.MarketRouterV2 = MagicMock()
main.market_router_v2.scan_markets.return_value = "BTCUSDT"

# Add missing globals
main.self_healer = MagicMock()
main.risk_engine_v3 = MagicMock()
main.risk_engine_v3.check_trade_risk.return_value = True # Allow trade
main.liquidity_ai = MagicMock()
main.liquidity_ai.analyze_intent.return_value = {'type': 'AGGRESSIVE'}
main.execution_quantum = MagicMock()
main.shadow_agent = MagicMock()

# Mock calculate_features to add dummy columns if the real one isn't working in this env?
# The real one uses 'ta'. Let's assume 'ta' is installed.
# But main.py imports it.
# We need to mock the functions main imports: add_ta_indicators etc.
# OR just rely on real ones.
# In main.py: from src.features.build_features import (add_lagged_features, ...)
# Since I mocked 'src.features.build_features' above, main.py has MOCKS for these functions.
# This means calculate_features won't do anything! 
# I need to UNMOCK them or provide a mocked implementation that adds the columns I need.

def mock_calc_features(df):
    # Add columns expected by main.py logic
    # "btc_rsi_14", "btc_bb_low", "btc_bb_high"
    # Logic in main: if rsi < 30 -> Buy.
    # We want to force a Buy.
    df["btc_rsi_14"] = 25.0
    df["btc_bb_low"] = 70000.0
    df["btc_bb_high"] = 80000.0
    df["btc_atr_14"] = 100.0
    return df

# Patch calculate_features in main module namespace
main.calculate_features = mock_calc_features

# Mock ML Models
main.ml_model_v2 = MagicMock()
main.ml_model_v2.predict_proba.return_value = [[0.2, 0.8]] # Strong Up
main.dot_model = MagicMock()
main.dot_model.return_value = (torch.tensor([0.9]), None) # Strong Up

# Run Job
print("Running main.job()...")
try:
    main.job()
    print("Job finished.")
except Exception as e:
    print(f"Job failed: {e}")
    import traceback
    traceback.print_exc()

