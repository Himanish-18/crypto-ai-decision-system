import logging
import pytz
import pickle
import pandas as pd
import numpy as np
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Any, List
from pathlib import Path

from src.features.alpha_signals import AlphaSignals
from src.risk_engine.regime_filter import RegimeFilter
from src.features.rl_signals import RLSignalEngine
from src.models.hybrid.tiny_cnn import TinyCNNProxy
from src.models.hybrid.tcn_lite import TCNLiteProxy
from src.models.hybrid.dqn_mini import DQNMiniProxy
from src.intelligence.noise_filter import NoiseFilter
from src.intelligence.kalman_smoother import KalmanSmoother
from src.intelligence.vol_adaptive import VolAdaptiveThreshold
from src.intelligence.trend_depth import TrendDepth
from src.features.sentiment_features import SentimentFeatures
from src.features.orderflow import OrderFlowFeatures
from src.models.panic_exit import PanicExitModel
from src.models.spread_cnn import SpreadCNN
from src.models.ppo_policy import PPOPolicy
import time
import json
from src.execution.hft_orderbook import HftOrderBook
from src.execution.fill_probability import FillProbabilityModel
from src.execution.transaction_cost_v2 import TransactionCostModelV2
from src.models.meta_regime_forecast import MetaRegimeForecast
from src.models.meta_label_safety import MetaLabelSafety
from src.execution.hedge_manager import HedgeManager
from src.intelligence.arbitrage_scanner import ArbitrageScanner
from src.data.deribit_vol_monitor import DeribitVolMonitor
from src.risk_engine.iv_guard import IVGuard

# Setup Logging
logger = logging.getLogger("live_engine")

class LiveSignalEngine:
    def __init__(self, model_path: Path, scaler_path: Path, balanced_mode: bool = True):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.balanced_mode = balanced_mode
        self.model = None
        self.scaler = None
        self.rl_engine = None # Lazy load
        
        # Hybrid v4 Models
        self.tiny_cnn = None
        self.tcn_lite = None
        self.dqn_mini = None
        
        self.regime_detector = None # Initialize regime_detector
        
        # v6 Intelligence Modules
        self.noise_filter = NoiseFilter(window=24, fft_threshold=0.20, var_threshold=0.0005)
        self.kalman = KalmanSmoother(process_noise=0.005, measurement_noise=0.03)
        self.vol_adaptive = VolAdaptiveThreshold(base_threshold=0.55, vol_sensitivity=0.15)
        self.trend_depth = TrendDepth()
        
        # v7 Components
        self.sentiment_gen = SentimentFeatures()
        from src.models.portfolio_rl.ppo_agent import PortfolioPPOAgent
        self.of_gen = OrderFlowFeatures()
        self.pem = PanicExitModel(sensitivity=1.0)
        self.spread_cnn = SpreadCNN()
        self.ppo_policy = PPOPolicy()
        self.hft_ob = HftOrderBook()
        self.fill_prob_model = FillProbabilityModel()
        self.tcm = TransactionCostModelV2()
        self.meta_regime = MetaRegimeForecast()
        self.meta_safety = MetaLabelSafety()
        self.hedge_manager = HedgeManager()
        self.arb_scanner = ArbitrageScanner()
        self.deribit = DeribitVolMonitor()
        self.iv_guard = IVGuard()
        self.portfolio_agent = PortfolioPPOAgent(input_dim=20, action_dim=5) # Init Agent
        from src.models.loss_prediction_model import LossPredictionModel
        from src.models.loss_prediction_model import LossPredictionModel
        self.loss_guard = LossPredictionModel()
        from src.risk_engine.correlation_guard import CorrelationGuard
        self.corr_guard = CorrelationGuard(window_size=30)
        from src.data.news_feed import NewsAggregator
        from src.models.news_classifier import NewsSentimentModel
        self.news_agg = NewsAggregator()
        self.news_model = NewsSentimentModel()
        
        self.cache_dir = self.model_path.parent.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_funding_rate = None
        self.last_funding_rate = None
        self.longs_disabled_until = pd.Timestamp.min # Timestamp (Naive)
        
        # v13 Low Latency State
        self.use_rust = False
        self.use_onnx = False
        try:
            import rust_engine
            self.rust = rust_engine
            self.use_rust = True
            logger.info("🦀 Rust Engine Loaded. Feature Engineering Accelerated.")
        except ImportError:
            logger.warning("⚠️ Rust Engine not found. Using Slow Python Pandas.")
            
        try:
            import onnxruntime as ort
            self.ort = ort
            self.use_onnx = True
            logger.info("🚀 ONNX Runtime Loaded. Inference Accelerated.")
        except ImportError:
            logger.warning("⚠️ ONNX Runtime not found. Using Slow PyTorch/TF.")
        
        # Online Feature Pruning State
        self.pruned_features = [] # Features to ignore
        
        # Hot-Swap State
        self.last_model_load_time = 0
        self.load_artifacts()
        
    def load_hybrid_models(self):
        root = self.model_path.parent / "hybrid"
        try:
            # v13 ONNX Loading
            if self.use_onnx and (root / "tiny_cnn_v2.onnx").exists():
                 self.tiny_cnn_ort = self.ort.InferenceSession(str(root / "tiny_cnn_v2.onnx"))
                 logger.info("🚀 Tiny-CNN Loaded via ONNX.")
            
            # v5 Models (Fallbacks / Hybrids)
            self.tiny_cnn = TinyCNNProxy.load(root / "tiny_cnn_v2.h5")
            self.tcn_lite = TCNLiteProxy.load(root / "tcn_lite_v2.h5")
            self.dqn_mini = DQNMiniProxy.load(root / "dqn_mini_v2.pt")
            
            # v5 Stacker
            import joblib
            if (root / "hybrid_v5_xgb.bin").exists():
                self.xgb_stacker = joblib.load(root / "hybrid_v5_xgb.bin")
                self.model = self.xgb_stacker # Alias for Guardian Compatibility
                logger.info("🧠 Hybrid v5 Models Loaded (Stacker Active).")
            else:
                self.xgb_stacker = None
                logger.warning("⚠️ v5 Stacker not found. Defaulting to fallback.")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Hybrid v5 Models: {e}. Falling back to standard.")

    def load_artifacts(self):
        """Load model and scaler."""
        # v5: self.model (Legacy Multifactor) is Deprecated.
        # We rely on Hybrid v5 Stacker loaded in load_hybrid_models.
        self.model = None
        # logger.info(f"📥 Loading model from {self.model_path}...")
        # with open(self.model_path, "rb") as f:
        #     self.model = pickle.load(f)
            
        logger.info(f"📥 Loading scaler from {self.scaler_path}...")
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
        # Load Regime Detector
        self.regime_model_path = self.model_path.parent / "regime_model.pkl"
        if self.regime_model_path.exists():
            logger.info(f"📥 Loading regime detector from {self.regime_model_path}...")
            import joblib
            self.regime_detector = joblib.load(self.regime_model_path)
        else:
            logger.warning("⚠️ Regime detector not found. Using default thresholds.")
            self.regime_detector = None

        # Load Selected Features Mask
        self.features_json_path = self.model_path.parent / "selected_alpha_features.json"
        self.selected_features = None
        
        if self.features_json_path.exists():
             logger.info(f"📥 Loading feature selection mask from {self.features_json_path}...")
             import json
             with open(self.features_json_path, "r") as f:
                 self.selected_features = json.load(f)
             logger.info(f"✅ Feature Mask Loaded: {len(self.selected_features)} features.")
        if self.ppo_policy.model_path and self.ppo_policy.model_path.exists():
             self.ppo_policy.load(self.ppo_policy.model_path)
             
        # Update Load Time
        if self.model_path.exists():
            self.last_model_load_time = self.model_path.stat().st_mtime
            
        # Load Hybrid Models (This sets self.model alias)
        self.load_hybrid_models()
            
        logger.info("✅ Hybrid v7 Models Loaded Successfully.")

    def check_for_model_updates(self):
        """
        Hot-Swap: Check if model file on disk is newer than loaded version.
        """
        if not self.model_path.exists(): return
        
        current_mtime = self.model_path.stat().st_mtime
        if current_mtime > self.last_model_load_time:
            logger.info("🔄 New Model Version Detected! Hot-Swapping...")
            try:
                self.load_artifacts()
                self.last_model_load_time = current_mtime
                logger.info("✅ Hot-Swap Complete. Active Logic Updated.")
            except Exception as e:
                logger.error(f"Hot-Swap Failed: {e}. Keeping old model.")

    def process_candle(self, candle_data: pd.DataFrame) -> Dict[str, Any]:
        # 0. Hot-Swap Check
        self.check_for_model_updates()
        
        """
        Process a single candle (row) to generate a signal.
        """
        # Drop non-feature cols
        exclude_cols = ["timestamp", "y_direction_up", "btc_ret_fwd_1", "y_pred", "y_prob", "signal_prob", "atr_pct", "is_shock", "is_uptrend", "signal_consistent", "entry_signal"]
        
        # 0. v7 Data Cache Snapshot
        try:
            snapshot_path = self.cache_dir / "latest_candle_snapshot.parquet"
            candle_data.tail(100).to_parquet(snapshot_path)
        except Exception as e:
            logger.warning(f"Cache Write Failed: {e}")

        # 1. v7 Feature Engineering (Sentiment + OrderFlow)
        candle_data = self.sentiment_gen.calculate_proxies(candle_data)
        candle_data = self.of_gen.calculate_features(candle_data)
        
        # 1.b v10 HFT Update (Simulated from Candle)
        # In real HFT, this is updated via WebSocket. Here we mock update to keep state alive.
        latest = candle_data.iloc[-1]
        self.hft_ob.update_snapshot(
            bids=[[latest["btc_close"] * 0.999, 1.0]], 
            asks=[[latest["btc_close"] * 1.001, 1.0]]
        )
        
        # 1.c v10 Meta-Regime Forecast
        meta_regime = self.meta_regime.predict(candle_data)
        logger.info(f"🔮 Meta-Regime: {meta_regime['predicted_regime']} ({meta_regime['confidence']:.2f})")

        # Online Feature Pruning Check (Simplified)
        # In real system, we'd reload mask here weekly
        
        # 1. Use Mask
        if self.selected_features:
            feature_cols = self.selected_features
        else:
            feature_cols = [c for c in candle_data.columns if c not in exclude_cols]
            
        # Regime Detection
        regime = "Unknown"
        risk_params = {}
        
        if self.regime_detector:
             try:
                 preds = self.regime_detector.predict(candle_data, symbol="btc")
                 raw_regime = preds.iloc[0]
                 ma50 = candle_data["btc_close"].rolling(50).mean().iloc[-1]
                 close = candle_data["btc_close"].iloc[-1]
                 
                 regime = "Sideways"
                 if raw_regime == "Trending":
                     if close > ma50: regime = "Bull Trend"
                     else: regime = "Bear Trend"
                 elif raw_regime == "HighVol":
                     regime = "High Volatility"
                     
                 # Get Risk Params
                 from src.risk_engine.regime_filter import RegimeFilter
                 rf_temp = RegimeFilter()
                 risk_params = rf_temp.get_risk_params(regime)
                 
             except Exception:
                 regime = "Sideways"
                 risk_params = {"entry_threshold": 0.55}
        
        # Context
        context = {"regime": regime, "risk_params": risk_params, "meta_regime": meta_regime}
        
        # Multi-Factor Prediction (Legacy Support)
        prob = 0.5
        mf_score = 0.5
        
        # Volatility
        volatility = 0.01
        if "btc_atr_14" in candle_data.columns:
             atr = float(candle_data["btc_atr_14"].iloc[-1])
             close = float(candle_data["btc_close"].iloc[-1])
             if close > 0: volatility = atr / close


        # --- BALANCED MODE LOGIC (Hybrid v5) ---
        if self.balanced_mode:
            # 1. Load Models if needed
            if not self.tiny_cnn or not self.tcn_lite:
                self.load_hybrid_models()
                
            # 2. Get Hybrid Scores
            cnn_score = 0.5
            tcn_score = 0.5
            dqn_q = 0.0
            
            if self.tiny_cnn: cnn_score = self.tiny_cnn.predict_score(candle_data)
            if self.tcn_lite: tcn_score = self.tcn_lite.predict_trend(candle_data.iloc[[-1]])
            
            # 3. v5 Stacker Inference
            prob = (cnn_score + tcn_score) / 2 # Fallback
            
            if hasattr(self, 'xgb_stacker') and self.xgb_stacker:
                # Prepare Stacker Inputs: [of_imbalance, vwap_z, smart_money, vol_rsi]
                # Assuming alpha features present in candle_data
                try:
                    stack_cols = ["btc_alpha_of_imbalance", "btc_alpha_vwap_zscore", "btc_alpha_smart_money_delta", 
                                  "btc_alpha_vol_flow_rsi"]
                    
                    vals = [float(latest.get(c, 0.0)) for c in stack_cols]
                    vals.append(cnn_score)
                    vals.append(tcn_score)
                    
                    X_stack = pd.DataFrame([vals], columns=stack_cols + ["cnn_prob", "tcn_prob"])
                    stacker_prob = self.xgb_stacker.predict_proba(X_stack)[0, 1]
                    logger.info(f"🧠 Hybrid v5 Stacker: {stacker_prob:.4f} (CNN={cnn_score:.2f}, TCN={tcn_score:.2f})")
                    prob = stacker_prob
                    mf_score = stacker_prob
                except Exception as e:
                    logger.error(f"Stacker Inference Error: {e}")

            # 4. v6 Intelligence Layer
            # A. Kalman Smoothing
            smooth_score = self.kalman.smooth(prob)
            logger.info(f"🌊 Kalman Smoothed: {prob:.4f} -> {smooth_score:.4f}")
            
            # --- v7 Panic Exit Check ---
            pem_out = self.pem.predict(candle_data)
            panic_score = pem_out["panic_score"]
            if pem_out["exit_signal"]:
                 logger.warning(f"🚨 PANIC EXIT TRIGGERED! Score: {panic_score:.2f} | Reason: {pem_out['reason']}")
                 return {
                    "timestamp": candle_data["timestamp"].iloc[-1],
                    "signal": -1, # Force Sell/Exit
                    "signal_confidence": panic_score,
                    "block_reason": f"PEM Panic: {pem_out['reason']}",
                    "execution_mode": "AGGRESSIVE", # Panic = Get out now
                    "strategy_context": context,
                    "prediction_prob": 0.0,
                    "ppo_size_scalar": 0.0
                 }
                 
            # --- v7 Funding Rate Logic ---
            if "fundingRate" in candle_data.columns:
                current_fund = candle_data["fundingRate"].iloc[-1]
                if self.last_funding_rate is not None:
                     # Check flip (sign change)
                     if np.sign(current_fund) != np.sign(self.last_funding_rate):
                          logger.warning(f"💸 Funding Rate Flip ({self.last_funding_rate} -> {current_fund}). Disabling Longs for 6 candles.")
                          # 6 candles * timeframe? Assume timestamp math or counter.
                          # Simplification: Disable for next 6 calls? 
                          # Store timestamp
                          current_ts = candle_data["timestamp"].iloc[-1]
                          # Assume 5m candles -> 30 mins
                          self.longs_disabled_until = current_ts + pd.Timedelta(minutes=30) # 6 * 5m
                self.last_funding_rate = current_fund
                
            # Check Disable
            current_ts = candle_data["timestamp"].iloc[-1]
            
            # Initialize with compatible type if 0
            if isinstance(self.longs_disabled_until, int) and self.longs_disabled_until == 0:
                 self.longs_disabled_until = pd.Timestamp.min.replace(tzinfo=current_ts.tzinfo)

            if current_ts < self.longs_disabled_until:
                 logger.info("🚫 Longs temporarily disabled due to Funding Flip.")
                 # If probabilistic signal is Long, we block it.
                 # If we were holding, we might not exit, but we don't enter.
                 # This function returns Signal 1 (Buy) or 0.
                 # We can just force 0.
                 return {
                    "timestamp": current_ts,
                    "signal": 0,
                    "block_reason": "Funding Flip Cooldown",
                    "strategy_context": context,
                    "prediction_prob": 0.0,
                    "execution_mode": "MAKER",
                    "ppo_size_scalar": 0.0
                 }

            # B. Noise/Chop Detection
            closes = candle_data["btc_close"]
            is_chop = self.noise_filter.is_chop(closes)
            is_chop = self.noise_filter.is_chop(closes)
            if is_chop:
                logger.warning("🌪️ Noise Filter: CHOP DETECTED. Requiring higher confidence (0.65).")

            # C. Loss Guard Veto (v7.1)
            # Feature Extraction for Loss Guard
            try:
                # 1h Return
                ret_1h = candle_data["close"].pct_change(12).iloc[-1] if len(candle_data) >= 12 else 0.0 # 12 * 5m = 1h
                # 4h Return
                ret_4h = candle_data["close"].pct_change(48).iloc[-1] if len(candle_data) >= 48 else 0.0
                # Skew (20 period)
                skew = candle_data["close"].rolling(20).skew().iloc[-1] if len(candle_data) >= 20 else 0.0
                # Spread Regime (Proxy: Volatility / Spread?) -> Simplified: Use Volatility
                vol_1h = candle_data["close"].pct_change().rolling(12).std().iloc[-1]
                # Funding Flip Bool
                funding_flip = 1 if (self.last_funding_rate and "fundingRate" in candle_data.columns and 
                                   np.sign(candle_data["fundingRate"].iloc[-1]) != np.sign(self.last_funding_rate)) else 0
                
                # Spread Regime (Dummy for now, or use Vol Adaptive logic)
                spread_regime = 1 if volatility > 0.01 else 0 # 1=HighVol, 0=LowVol
                
                loss_feats = {
                    "ret_1h": ret_1h,
                    "ret_4h": ret_4h,
                    "vol_1h": vol_1h,
                    "skew": skew,
                    "funding_flip": funding_flip,
                    "spread_regime": spread_regime
                }
                
                veto, loss_prob = self.loss_guard.check_veto(loss_feats)
                if veto:
                     logger.warning(f"🛡️ LossGuard VETO! P(Loss) {loss_prob:.2f} > 0.6. Blocking Trade.")
                     return {
                        "timestamp": current_ts,
                        "signal": 0,
                        "block_reason": f"LossGuard Veto (P={loss_prob:.2f})",
                        "strategy_context": context,
                        "prediction_prob": 0.0,
                        "execution_mode": "MAKER",
                        "ppo_size_scalar": 0.0
                     }
            except Exception as e:
                logger.error(f"LossGuard Error: {e}")
                # Fail open (continue) or fail closed? Fail Open usually better for aux safety unless critical.
                pass

            # C. Adaptive Threshold
            adaptive_thresh = self.vol_adaptive.get_threshold(volatility)
            
            # D. Trend Depth (v6.1)
            td_score = self.trend_depth.calculate(candle_data)
            logger.info(f"🌊 Trend Depth: {td_score:.4f}")

            # C. Loss Guard Veto (v7.1) -- (Existing Block)
            
            # D. Correlation Guard (v7.2)
            # Update Guard with latest prices
            # TODO: Get real ETH/SOL/LTC prices from MarketRouter
            # For now, we simulate correlated data to enable logic testing
            btc_price = candle_data["close"].iloc[-1]
            # Mock Peers (Correlated random walk)
            # In production, pass real dict from engine input
            mock_prices = {
                "BTC": btc_price,
                "ETH": btc_price * 0.05 + np.random.normal(0, 10), # Correlated
                "SOL": btc_price * 0.001 + np.random.normal(0, 5),
                "LTC": btc_price * 0.002 + np.random.normal(0, 2)
            }
            self.corr_guard.update(mock_prices)
            
            # Get Modifiers
            # Use TCN score as proxy for current size intent (0.0 to 1.0)
            corr_scalar, hedge_req, corr_debug = self.corr_guard.calculate_risk_modifiers(current_pos_size=tcn_score)
            
            if corr_scalar < 1.0:
                logger.info(f"🔗 Correlation scalar applied: {corr_scalar:.2f}")

            
            # D. Correlation Guard (v7.2) -- (Existing Block)
            # ...
            # E. News Sentiment Guard (v7.3)
            try:
                # 1. Fetch Headlines
                headlines = self.news_agg.fetch_headlines()
                # 2. Classify
                sentiment_score = self.news_model.predict(headlines)
                
                # Check Risk Condition: Bearish Sentiment (< -0.5) AND High Volatility
                # Volatility threshold: e.g., 1.5x average ATR or > 1% move?
                # Using 'vol_1h' if available, or simple ATR check
                vol_metric = candle_data["btc_atr_14"].iloc[-1] / candle_data["btc_close"].iloc[-1] if "btc_atr_14" in candle_data.columns else 0.01
                
                if sentiment_score < -0.5 and vol_metric > 0.005: # 0.5% ATR ~ High Vol
                    logger.warning(f"📰 News Risk: Sentiment {sentiment_score:.2f} (Bearish) + Vol {vol_metric:.4f}. Blocking Longs.")
                    # Return Block Signal
                    return {
                        "timestamp": current_ts,
                        "signal": 0,
                        "block_reason": f"News Risk (Sent={sentiment_score:.2f})",
                        "strategy_context": context,
                        "prediction_prob": 0.0,
                        "execution_mode": "MAKER",
                        "ppo_size_scalar": 0.0
                    }
                elif sentiment_score < -0.1:
                    logger.info(f"📰 News Sentiment: {sentiment_score:.2f} (Bearish). Caution advised.")
                
            except Exception as e:
                logger.error(f"News Guard Error: {e}")
            
            # 5. DQN Policy Check
            if self.dqn_mini:
                # Use SMOOTHED score for DQN as requested
                dqn_q = self.dqn_mini.predict_q_value(
                    candle_data.iloc[-1],
                    smooth_score, 
                    cnn_score, 
                    tcn_score
                )
                logger.info(f"🤖 DQN Policy Q-Value: {dqn_q:.6f}")
                
                # VETO LOGIC
                if dqn_q <= 0.0:
                     logger.info(f"⛔ DQN VETO: Q-Value {dqn_q:.6f} <= 0. Signal Blocked.")
                     return {
                        "timestamp": candle_data["timestamp"].iloc[-1],
                        "signal": 0,
                        "signal_confidence": 0.0,
                        "block_reason": "DQN Veto",
                        "strategy_context": context,
                        "prediction_prob": float(smooth_score),
                        "execution_mode": "MAKER",
                        "ppo_size_scalar": 0.0
                     }

        # v7 Sideways Micro-Scalp Mode
        # If Volatility is VERY low < 0.00045 (~0.045%)
        # Standard logic might block due to Chop Filter.
        # But User requested "Switch to Sideways Micro-Scalp Mode".
        
        is_micro_scalp = False
        if volatility < 0.00045:
             logger.info(f"🦀 Sideways Micro-Scalp Mode Active (Vol {volatility:.6f} < 0.00045)")
             is_micro_scalp = True
             # Override Chop Block?
             # Logic: RSI + VWAP Mean Reversion.
             # If Price < VWAP and RSI < 30 -> Buy
             # This is a separate signal source.
             
             # Calculate RSI & VWAP deviation
             rsi = 50
             if "rsi" in candle_data.columns: rsi = candle_data["rsi"].iloc[-1]
             
             price = candle_data["btc_close"].iloc[-1]
             vwap = candle_data["btc_close"].ewm(span=20).mean().iloc[-1] # Proxy
             
             if rsi < 35 and price < vwap:
                  logger.info("🦀 Micro-Scalp Long Signal (RSI < 35 & Price < VWAP)")
                  final_signal = 1
                  effective_threshold = 0.0 # Force entry
                  execution_mode = "MAKER"
                  # Final Decision Logic (v6.1 + v7)
        # 1. Base Logic
        final_signal = 0
        block_reason = None
        execution_mode = "MAKER"
        
        # v10 Execution Logic: Fill Probability
        # Determine Execution Mode based on Fill Prob
        # Assume we want to buy at best bid?
        imb = self.hft_ob.get_imbalance()
        fill_prob = self.fill_prob_model.estimate_fill_prob("buy", 0, 0, imb, volatility)
        logger.info(f"⚡ Fill Prob: {fill_prob:.2f}")
        
        if fill_prob > 0.6:
            execution_mode = "MAKER"
        else:
            execution_mode = "TAKER"
        
        # 2. Trend Depth Modifiers (v6.1 Recalibrated)
        effective_threshold = adaptive_thresh
        if not is_micro_scalp: # Only apply standard TD checks if not in micro mode
            if td_score > 0.65:
                # Strong Trend: Lower barrier, possibly go Taker
                effective_threshold -= 0.05
                logger.info(f"📉 Trend Boost: Threshold lowered to {effective_threshold:.4f}")
                
            elif td_score < 0.25:
                 # Weak Trend: Raise barrier, force Maker
                 effective_threshold += 0.04
                 logger.info(f"🛡️ Weak Trend Protection: Threshold raised to {effective_threshold:.4f}")
                 execution_mode = "MAKER"
        # 3. Chop Handling with Override
        if is_chop:
            if td_score > 0.75:
                # Override Chop if Trend Depth is VERY high (Breakout)
                logger.info("⚔️ CHOP OVERRIDE: Strong Trend Depth (>0.75) detected.")
                if smooth_score > effective_threshold: 
                    final_signal = 1
            else:
                 # Standard Chop Block
                 if smooth_score > 0.65:
                    logger.info("⚔️ CHOP FILTER PASSED: High Confidence.")
                    final_signal = 1
                 else:
                     logger.info("🛡️ Blocked by Noise Filter (Score < 0.65 in Chop).")
                     block_reason = "Noise Filter"
        else:
            # Normal Regime
            if smooth_score > effective_threshold:
                final_signal = 1
            else:
                logger.info(f"💤 Score {smooth_score:.4f} <= Eff. Threshold {effective_threshold:.4f}")
        
        # v10 Meta-Label Safety Check
        if final_signal == 1:
            # Check v10 Meta Safety
            safe_context = {"signal_confidence": float(smooth_score - effective_threshold)}
            if not self.meta_safety.check_safety(safe_context):
                final_signal = 0
                block_reason = "Meta-Label Safety Veto"
                logger.warning("🛡️ Signal VETOED by Meta-Label Safety Model.")

        # v10 TCM Cost Check
        if final_signal == 1:
            est_cost = self.tcm.estimate_cost("buy", 1000, volatility, 0.0001, mode=execution_mode) # Dummy spread
            if not self.tcm.is_profitable(0.002, est_cost): # Require 0.2% raw return? Heuristic.
                 logger.warning(f"💸 TCM VETO: Expected Return < Cost ({est_cost*100:.3f}%). Blocked.")
                 final_signal = 0
                 block_reason = "Transaction Cost Veto"

        # v7 PPO Sizing
        size_scalar = 0.0
        if final_signal == 1:
            ppo_state = {
                "prob": float(smooth_score),
                "volatility": float(volatility),
                "trend_depth": float(td_score),
                "panic_score": float(panic_score)
            }
            size_scalar = self.ppo_policy.get_action(ppo_state)
            logger.info(f"🧠 PPO Sizing: {size_scalar:.2f}x (Prob={smooth_score:.2f}, Vol={volatility:.4f})")
            
            # If PPO says 0 size, we block
            if size_scalar < 0.1:
                logger.warning(f"🧠 PPO Veto: Size {size_scalar:.2f} < 0.1. Signal Blocked.")
                final_signal = 0
                block_reason = "PPO Sizing Veto"

        if final_signal == 1:
            logger.info(f"✅ SIGNAL GENERATED: Score {smooth_score:.4f} | Size: {size_scalar:.2f}x | Exec: {execution_mode}")
            return {
                "timestamp": candle_data["timestamp"].iloc[-1],
                "prediction_prob": float(smooth_score),
                "signal": 1, 
                "signal_confidence": float(smooth_score - effective_threshold),
                "strategy_context": context,
                "execution_mode": execution_mode,
                "ppo_size_scalar": float(size_scalar)
            }
        else:
            return {
                "timestamp": candle_data["timestamp"].iloc[-1],
                "signal": 0,
                "prediction_prob": float(smooth_score),
                "strategy_context": context,
                "block_reason": block_reason,
                "execution_mode": execution_mode,
                "ppo_size_scalar": 0.0
            }

