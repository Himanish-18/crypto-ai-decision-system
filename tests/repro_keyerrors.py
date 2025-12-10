
import pandas as pd
import numpy as np
import logging
import sys
from unittest.mock import MagicMock, patch

# Mocking modules that might be hard to load or missing
sys.modules['src.data.deribit_vol_monitor'] = MagicMock()
sys.modules['src.risk_engine.iv_guard'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['src.models.portfolio_rl.ppo_agent'] = MagicMock()

# Setup Logging
logging.basicConfig(level=logging.INFO)

try:
    from src.execution.live_signal_engine import LiveSignalEngine
except ImportError:
    # Adjust path if needed (running from root)
    import os
    sys.path.append(os.getcwd())
    from src.execution.live_signal_engine import LiveSignalEngine

def test_fixes():
    print("Testing Fixes...")
    
    # Mock paths
    mock_model_path = MagicMock()
    mock_model_path.parent.parent = MagicMock()
    mock_model_path.exists.return_value = False
    
    mock_scaler_path = MagicMock()
    
    # Initialize Engine (Mocking heavy inits if necessary, but trying real first)
    # We might need to mock __init__ parts if they load heavy models, but let's try.
    # To avoid loading real heavy models, we can patch load_artifacts
    
    with patch.object(LiveSignalEngine, 'load_artifacts') as mock_load:
        with patch.object(LiveSignalEngine, 'load_hybrid_models') as mock_load_hybrid:
            engine = LiveSignalEngine(mock_model_path, mock_scaler_path)
            
            # Manually init simplified components needed for crash test
            engine.sentiment_gen = MagicMock()
            # Define calculating proxies to mimic the bug scenario (but we fixed it, so we want to see it work)
            # Actually, we want to test the REAL sentiment_gen method if possible.
            from src.features.sentiment_features import SentimentFeatures
            engine.sentiment_gen = SentimentFeatures()
            engine.selected_features = None # Fix AttributeError
            
            from src.execution.hft_orderbook import HftOrderBook
            engine.hft_ob = HftOrderBook()
            
            from src.models.meta_regime_forecast import MetaRegimeForecast
            engine.meta_regime = MagicMock()
            engine.meta_regime.predict.return_value = {"predicted_regime": "STABLE", "confidence": 0.5}
            
            engine.fill_prob_model = MagicMock()
            engine.fill_prob_model.estimate_fill_prob.return_value = 0.8
            
            engine.tcm = MagicMock()
            engine.tcm.estimate_cost.return_value = 0.0001
            engine.tcm.is_profitable.return_value = True
            
            engine.meta_safety = MagicMock()
            engine.meta_safety.check_safety.return_value = True
            
            engine.ppo_policy = MagicMock()
            engine.ppo_policy.get_action.return_value = 1.0

            engine.kalman = MagicMock()
            engine.kalman.smooth.return_value = 0.6 # Signal
            
            engine.pem = MagicMock()
            engine.pem.predict.return_value = {"exit_signal": False, "panic_score": 0.0}

            engine.of_gen = MagicMock()
            engine.of_gen.calculate_features.side_effect = lambda x: x # Pass through
            
            engine.vol_adaptive = MagicMock()
            engine.vol_adaptive.get_threshold.return_value = 0.5
            
            engine.trend_depth = MagicMock()
            engine.trend_depth.calculate.return_value = 0.5
            
            # --- Test Case 1: Missing 'volume' column (KeyError Fix) ---
            print("\n--- Test 1: Missing 'volume' (using btc_volume) ---")
            df = pd.DataFrame({
                "timestamp": [pd.Timestamp.now()],
                "btc_close": [60000.0],
                "btc_volume": [100.0],
                "fundingRate": [0.0001],
                "openInterest": [1000.0],
                "high": [60100], "low": [59900], "close": [60000], "open": [60000] # For spread calc
            })
            
            # Run
            try:
                # The sentiment_gen should now handle btc_volume
                res = engine.sentiment_gen.calculate_proxies(df)
                print("✅ Sentiment Features Calculated (Volume Fix Works)")
            except KeyError as e:
                print(f"❌ Failed: KeyError {e}")

            # --- Test 2: TCN Lite Missing Features ---
            print("\n--- Test 2: TCN Lite Missing Features ---")
            from src.models.hybrid.tcn_lite import TCNLiteProxy
            tcn = TCNLiteProxy()
            tcn.is_fitted = True
            tcn.feature_cols = ["missing_feat_1", "missing_feat_2"]
            tcn.model = MagicMock()
            tcn.model.predict_proba.return_value = np.array([[0.0, 0.6]])
            
            # Row without those features
            row = pd.DataFrame({"other_feat": [1]})
            
            try:
                prob = tcn.predict_trend(row)
                print(f"✅ TCN Prediction: {prob} (Reindexing Fix Works)")
            except KeyError as e:
                print(f"❌ Failed: KeyError {e}")
                
            # --- Test 3: LiveSignalEngine Return Keys (Funding Flip / Early Exits / Normal) ---
            print("\n--- Test 3: Return Dictionary Keys ---")
            engine.tcn_lite = tcn
            engine.tiny_cnn = MagicMock()
            engine.tiny_cnn.predict_score.return_value = 0.5
            
            # Trigger Funding Flip
            engine.last_funding_rate = -0.0001
            df_flip = df.copy()
            df_flip["fundingRate"] = 0.0001 # Flip sign
            
            res = engine.process_candle(df_flip)
            required_keys = ["strategy_context", "prediction_prob", "execution_mode"]
            missing = [k for k in required_keys if k not in res]
            if not missing:
                print("✅ Funding Flip Return keys present")
            else:
                print(f"❌ Funding Flip Missing keys: {missing}")

            # Trigger PEM Panic
            engine.pem.predict.return_value = {"exit_signal": True, "panic_score": 0.9, "reason": "Test"}
            res = engine.process_candle(df)
            missing = [k for k in required_keys if k not in res]
            if not missing:
                print("✅ Panic Return keys present")
            else:
                 print(f"❌ Panic Missing keys: {missing}")

            # Trigger DQN Veto
            engine.pem.predict.return_value = {"exit_signal": False, "panic_score": 0.0}
            engine.dqn_mini = MagicMock()
            engine.dqn_mini.predict_q_value.return_value = -0.1 # Veto
            
            res = engine.process_candle(df)
            missing = [k for k in required_keys if k not in res]
            if not missing:
                print("✅ DQN Veto Return keys present")
            else:
                 print(f"❌ DQN Veto Missing keys: {missing}")

if __name__ == "__main__":
    test_fixes()
