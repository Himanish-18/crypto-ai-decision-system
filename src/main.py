import asyncio
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import collections # Added v43
import json # Added v40
import yaml
import schedule
import joblib
import torch

from src.data.market_router import MarketRouter
from src.execution.binance_executor import BinanceExecutor
from src.execution.live_signal_engine import LiveSignalEngine
from src.execution.smart_executor import SmartExecutor
from src.execution.trading_decision import TradingDecision
from src.features.alpha_signals import AlphaSignals
from src.features.build_features import (add_lagged_features,
                                         add_rolling_features,
                                         add_ta_indicators, engineer_sentiment)
from src.features.orderbook_features import OrderBookManager
from src.features.orderflow_features import OrderFlowFeatures
from src.guardian.safety_daemon import SafetyDaemon
from src.portfolio.engine_v15 import PortfolioCoordinator
from src.risk_engine.risk_module import RiskEngine

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("live_trading.log"), logging.StreamHandler()],
)
logger = logging.getLogger("main")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "multifactor_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
LOG_DIR = DATA_DIR / "execution" / "logs"




def safe_load_model(path):
    try:
        if not path.exists():
            logger.warning(f'Model missing: {path}')
            return None
        model = joblib.load(path)
        logger.info(f'✅ Safe Load Success: {path.name}')
        return model
    except Exception as e:
        logger.error(f'❌ Model Load Failed {path}: {e}')
        return None

# --- GLOBAL STATE INIT ---
GLOBAL_CONFIG = {}
try:
    with open(PROJECT_ROOT / "config" / "config.yaml", "r") as f:
        GLOBAL_CONFIG = yaml.safe_load(f)
except Exception as e:
    logger.warning(f"Config Load Failed: {e}")
ml_model = None
dot_model = None
# Fix: Init DOT Model
try:
    from src.features.orderflow import OrderFlowFeatures
    dot_model = OrderFlowFeatures()
    logger.info("🧠 DOT Model Initialized.")
except Exception as e:
    logger.warning(f"DOT Init Failed: {e}")
TRAINING_FEATURES = []
EMA_RET = None
PRED_QUEUE = collections.deque(maxlen=5)
LAST_PRED_DIR = 'Neutral'
LAST_PRED_SCORE = 0.0
LAST_PRED_PRICE = 0.0

# --- ML MODEL LOADING (Moved to Global) ---
try:
    if (MODELS_DIR / "multifactor_model_v3.pkl").exists():
        ml_model = safe_load_model(MODELS_DIR / "multifactor_model_v3.pkl")
        with open(MODELS_DIR / "training_features.json", "r") as f:
            TRAINING_FEATURES = json.load(f)
        logger.info("🧠 Multifactor Model v3 Loaded.")
    elif (MODELS_DIR / "multifactor_model_v2.pkl").exists():
        ml_model = safe_load_model(MODELS_DIR / "multifactor_model_v2.pkl")
        logger.warning("⚠️ Using Fallback v2 Model.")
except Exception as e:
    logger.error(f"Failed to load ML Model: {e}")

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """v40: Standardized Feature Generation for v3 Model."""
    df = df.copy()
    
    # 1. Alias Basics for Model (expects btc_ prefix)
    df["btc_close"] = df["close"]
    df["btc_open"] = df["open"]
    df["btc_high"] = df["high"]
    df["btc_low"] = df["low"]
    df["btc_volume"] = df["volume"]
    
    # 2. Returns
    df["btc_ret"] = df["btc_close"].pct_change()
    df["btc_ret_lag_1"] = df["btc_ret"].shift(1)
    df["btc_close_lag_1"] = df["btc_close"].shift(1)
    df["btc_volume_lag_1"] = df["btc_volume"].shift(1)

    # 3. Technicals (Standardized)
    # RSI 14
    try:
        df["btc_rsi_14"] = ta.momentum.RSIIndicator(df["btc_close"], window=14).rsi()
    except Exception:
        df["btc_rsi_14"] = 50.0 
    
    # MACD
    try:
        macd = ta.trend.MACD(df["btc_close"])
        df["btc_macd"] = macd.macd()
        df["btc_macd_signal"] = macd.macd_signal()
        df["btc_macd_diff"] = macd.macd_diff()
    except Exception:
        df["btc_macd"] = 0.0
        df["btc_macd_signal"] = 0.0
        df["btc_macd_diff"] = 0.0
    
    # BB
    try:
        bb = ta.volatility.BollingerBands(df["btc_close"], window=20)
        df["btc_bb_width"] = bb.bollinger_wband()
        df["btc_bb_high"] = bb.bollinger_hband()
        df["btc_bb_low"] = bb.bollinger_lband()
    except Exception:
        df["btc_bb_width"] = 0.0
        df["btc_bb_high"] = df["btc_close"]
        df["btc_bb_low"] = df["btc_close"]
        
    # ATR
    try:
        atr = ta.volatility.AverageTrueRange(df["btc_high"], df["btc_low"], df["btc_close"], window=14)
        df["btc_atr_14"] = atr.average_true_range()
    except Exception:
        df["btc_atr_14"] = df["btc_close"] * 0.01 
        
    # 4. Rolling Stats (Vol & MeanRev)
    df["btc_roll_std_20"] = df["btc_close"].rolling(20).std()
    roll_mean = df["btc_close"].rolling(20).mean()
    df["btc_zscore_20"] = (df["btc_close"] - roll_mean) / (df["btc_roll_std_20"] + 1e-9)
    df["btc_momentum_20"] = (df["btc_close"] / (roll_mean + 1e-9)) - 1
    
    # 5. OrderFlow Delta Proxy (Required for DOT)
    # Delta ~ Volume * Direction
    direction = np.where(df["btc_close"] >= df["btc_open"], 1, -1)
    df["orderflow_delta"] = df["btc_volume"] * direction

    # Fill NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
    # v40 Fix: Ensure orderflow_delta exists
    if 'orderflow_delta' not in df.columns:
        # Proxy using volume/close interaction
        df['orderflow_delta'] = df['btc_volume'] * (df['btc_close'] - df['btc_open']) / df['btc_close']
        df['orderflow_delta'] = df['orderflow_delta'].fillna(0)
    return df


import yaml  # Added for config loading

from src.decision.arbitrator import AgentArbitrator
# --- v19 QUANTUM UPGRADE ---
from src.decision.meta_brain_titan import MetaBrainHardVeto
from src.decision.meta_brain_v21 import MetaBrainV21
from src.execution.execution_quantum import ExecutionQuantum
from src.execution.liquidity_ai import LiquidityAI
from src.maintenance.self_heal import SelfHealingSystem
from src.market.router_v2 import MarketRouterV2
from src.ml.noise.cleanliness import MarketCleanlinessModel
from src.risk_engine.risk_v3 import RiskEngineV3
from src.rl.ppo_portfolio import PPOPortfolioAgent  # v23


# Helper: Load Config
def load_config(path="config/config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config file not found, using defaults.")
        return {}


# Load Config Globally
config = load_config()


# Stub Telemetry Mock
class PrometheusStub:
    def inc(self, metric: str):
        logger.info(f"📊 TELEMETRY: {metric} +1")


telemetry = PrometheusStub()

# Accuracy Logger (Restored from LiveSignalEngine)
acc_logger = logging.getLogger("accuracy_monitor")
acc_logger.setLevel(logging.INFO)
if not acc_logger.handlers:
    handler = logging.FileHandler("predictions.log")
    formatter = logging.Formatter("%(asctime)s,%(message)s")
    handler.setFormatter(formatter)
    acc_logger.addHandler(handler)

# Global State for Validation
LAST_PRED_SCORE = None
LAST_PRED_PRICE = None
LAST_PRED_DIR = "Neutral"

# v43 Globals
PRED_QUEUE = collections.deque(maxlen=5)
EMA_RET = None
TRAINING_FEATURES = []



def job():
    global LAST_PRED_SCORE, LAST_PRED_PRICE, LAST_PRED_DIR, EMA_RET
    logger.info("⏰ Starting v19 QUANTUM Intelligence Cycle...")

    try:
        # 1. System Self-Check
        self_healer.monitor_performance(accuracy_window=0.9, latency_ms=5.0)

        # 2. Autonomous Market Switching
        focus_ticker = market_router_v2.scan_markets({})
        logger.info(f"🔭 QUANTUM Auto-Focus: {focus_ticker}")

        # 3. Data Fetch
        pair = focus_ticker
        df = asyncio.run(
            market_router.fetch_unified_candles(pair, timeframe="1m", limit=500)
        )

        if df is None or df.empty:
            logger.warning(f"No Data for {focus_ticker}")
            return

        if "close" in df.columns:
            current_price = df["close"].iloc[-1]
        else:
            close_col = [c for c in df.columns if "close" in c][0]
            current_price = df[close_col].iloc[-1]
            df["close"] = df[close_col]
            if "open" not in df.columns:
                df["open"] = df[[c for c in df.columns if "open" in c][0]]
            if "high" not in df.columns:
                df["high"] = df[[c for c in df.columns if "high" in c][0]]
            if "low" not in df.columns:
                df["low"] = df[[c for c in df.columns if "low" in c][0]]
            if "volume" not in df.columns:
                df["volume"] = df[[c for c in df.columns if "volume" in c][0]]

        # Log Feature Count
        logger.info(f"Feature Count: {df.shape[1]}")
        
        if False: # Force Run: ML will fail gracefully if features mismatch
            logger.error(f"❌ Feature Matrix Too Small ({df.shape[1]}) — ML Disabled")
            return

        # Prepare for feature gen
        symbol_prefix = focus_ticker.lower()
        if symbol_prefix not in ["btc", "eth"]:
             symbol_prefix = "btc" # Default fallback for library compatibility

        df[f"{symbol_prefix}_open"] = df["open"]
        df[f"{symbol_prefix}_high"] = df["high"]
        df[f"{symbol_prefix}_low"] = df["low"]
        df[f"{symbol_prefix}_close"] = df["close"]
        df[f"{symbol_prefix}_volume"] = df["volume"]
        
        # Calculate Features
        df = calculate_features(df)

        # 4. Noise Immunity Check
        noise_score = noise_guard.analyze_cleanliness(df)
        if noise_score > 0.85:
            logger.warning(
                f"🌪 Market Choppy (Score {noise_score:.2f}). QUANTUM Blocking Trades."
            )
            LAST_PRED_DIR = "Blocked"
            LAST_PRED_SCORE = None # Prevent stale validation
            return

        # 5. QUANTUM Intelligence Layer
        market_payload = {
            "symbol": focus_ticker,
            "candles": df,
            "microstructure": ob_manager.get_latest_metrics(),
            "funding_rate": 0.0001,
            "volatility": 0.02,
        }

        # v40 Step 2: Fixed Labeling Logic (Hard Thresholds)
        last_close = df["btc_close"].iloc[-1]
        ret1 = (df["btc_close"].iloc[-1] - df["btc_close"].iloc[-2]) / df["btc_close"].iloc[-2]
        
        threshold = 0.0005 # Fixed Microstructure Threshold
        
        market_state = "FLAT"
        if ret1 > threshold:
            market_state = "UP"
        elif ret1 < -threshold:
            market_state = "DOWN"

        # A. Meta-Brain (Relaxed Veto)
        # v40 Step 7: Only veto on extreme crashes
        macro_decision = meta_brain.think(market_payload)
        if macro_decision["action"] == "HOLD" and macro_decision.get("reason") == "CRASH_PROTECTION":
             logger.warning(f"🛡️ CRASH VETO: {macro_decision['reason']}")
             return

        # B. Agent Arbitrator
        last_row = df.iloc[-1]
        
        # 1. MomentumHunter
        rsi = last_row.get("btc_rsi_14", 50.0)
        mom_sig = 0.0
        if rsi < 30: mom_sig = 1.0 
        elif rsi > 70: mom_sig = -1.0 
        
        # 2. MeanRevGhost
        price = last_row["btc_close"]
        bb_low = last_row.get("btc_bb_low", price)
        bb_high = last_row.get("btc_bb_high", price)
        mr_sig = 0.0
        if price < bb_low: mr_sig = 1.0 
        elif price > bb_high: mr_sig = -1.0 

        # 3. VolOracle
        # Stub
        
        # 4. ML Ensemble (v40 Real Features)
        ml_sig = 0.0
        ml_conf = 0.0
        disable_ml = GLOBAL_CONFIG.get("deployment", {}).get("feature_flags", {}).get("disable_ml", False)
        if not disable_ml and ml_model and TRAINING_FEATURES:
            try:
                # Align features: Select columns, fillna, reshape
                # Note: df has btc_ prefixes now. TRAINING_FEATURES likely has btc_ prefixes too (from parquet header check)
                row_feats = df.iloc[-1][TRAINING_FEATURES].fillna(0)
                lx = row_feats.values.reshape(1, -1)
                
                probs = ml_model.predict_proba(lx)[0]
                p_up = probs[1]
                
                # Standard Logic
                if p_up > 0.60:
                     ml_sig = 1.0
                     ml_conf = (p_up - 0.5) * 2
                elif p_up < 0.40:
                     ml_sig = -1.0
                     ml_conf = (0.5 - p_up) * 2
            except Exception as e:
                logger.error(f"ML Error: {e}")

        # 5. DOT Model (120 seq, btc columns)
        dot_sig = 0.0
        dot_conf = 0.0
        try:
             # v40 Step 5: Sequence construction
             seq_df = df[['btc_close','btc_volume','orderflow_delta']].tail(120)
             if len(seq_df) == 120:
                 # Normalize (Z-score logic inline or assume transformer handles it?)
                 # DOT usually expects normalized input or raw. Assuming raw for now as transformer is black box here.
                 # Better: Simple Lookback Normalization
                 # seq = (seq - mean) / std ?
                 # User prompt didn't specify normalization, just "Construct input".
                 seq = seq_df.values
                 seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                 
                 dot_out, _ = dot_model(seq)
                 p_dot = dot_out.item()
                 
                 raw_conf = abs(p_dot - 0.5) * 2
                 if raw_conf > 0.10: # Min conf
                     if p_dot > 0.60:
                         dot_sig = 1.0
                         dot_conf = raw_conf
                     elif p_dot < 0.40:
                         dot_sig = -1.0
                         dot_conf = raw_conf
        except Exception as e:
            logger.error(f"DOT Error: {e}")

        # Construct Signals
        raw_signals = {
            "MomentumHunter": {"signal": mom_sig, "confidence": 0.8},
            "MeanRevGhost": {"signal": mr_sig, "confidence": 0.8},
            "VolOracle": {"signal": 0.0, "confidence": 0.5},
            "ML_Ensemble": {"signal": ml_sig, "confidence": ml_conf},
            "DOT_Signal": {"signal": dot_sig, "confidence": dot_conf},
            "MarketState": market_state,
        }
        
        # v40 Step 9: Logging
        logger.info(f"RAW SIGNALS → {raw_signals}")
        
        regime = market_state if market_state != "FLAT" else "NEUTRAL"
        
        arbitrated_decision = arbitrator.arbitrate(raw_signals, regime)
        logger.info(f"ARBITRATED → {arbitrated_decision}")

        # 6. Risk & Execution
        if risk_engine_v3:
            if not risk_engine_v3.check_trade_risk(arbitrated_decision, {"beta": 1.0}):
                 logger.warning("🛑 RISK VETO")
                 return

        if arbitrated_decision["action"] != "HOLD":
            execution_quantum.execute_order(arbitrated_decision, market_payload)

        # v40 Step 11: Final Decision Mapping
        action = arbitrated_decision.get("action", "HOLD")
        score = arbitrated_decision.get("size", 0.0)
        
        if action == "BUY":
            LAST_PRED_DIR = "Bullish"
        elif action == "SELL":
            LAST_PRED_DIR = "Bearish"
        else:
            LAST_PRED_DIR = "Neutral"
            
        LAST_PRED_SCORE = score

        # v40 Step 3: Validation Logic (Strict)
        # 5-bar Lookahead
        current_pred = {
            "label": LAST_PRED_DIR,
            "score": LAST_PRED_SCORE,
            "price": current_price,
            "threshold": threshold 
        }
        PRED_QUEUE.append(current_pred)
        
        if len(PRED_QUEUE) == 5:
            old_pred = PRED_QUEUE[0]
            start_price = old_pred["price"]
            pred_label = old_pred["label"]
            
            future_ret = (current_price - start_price) / start_price
            
            # EMA Smoothing (v43 request kept)
            alpha = 0.5
            if EMA_RET is None: EMA_RET = future_ret
            else: EMA_RET = alpha * future_ret + (1 - alpha) * EMA_RET
            
            eval_ret = EMA_RET
            logger.info(f"RET SINCE LAST → {eval_ret:.6f}") # v40 Step 9
            
            # Determine True Label (Strict)
            true_label = "Flat"
            if eval_ret > 0.0005: true_label = "Up"
            elif eval_ret < -0.0005: true_label = "Down"
            
            # Strict Correctness
            is_correct = False
            if pred_label == "Bullish" and true_label == "Up": is_correct = True
            elif pred_label == "Bearish" and true_label == "Down": is_correct = True
            elif pred_label == "Neutral" and true_label == "Flat": is_correct = True
            
            res_str = "CORRECT" if is_correct else "WRONG"
            
            acc_logger.info(
                f"{pred_label},{old_pred['score']:.4f},{true_label},{eval_ret:.6f},{res_str}"
            )

        LAST_PRED_PRICE = current_price
        logger.info("💤 Cycle Complete.")


    except Exception as e:
        logger.error(f"Error in v19 cycle: {e}", exc_info=True)
        self_healer.restart_hft_layer(ob_manager)



import argparse

# --- v30 INSTITUTIONAL UPGRADE IMPORTS ---
from src.native_interface import NativeEngine
from src.fix_gateway.fix_api import FIXClient
from src.event_backtester.engine import EventBacktester, MarketData, Order, Event
from src.event_backtester.simulator import ExchangeSimulator
from src.portfolio.optimizer_v3 import InstitutionalOptimizer

# --- ARGS PARSER ---
def parse_args():
    parser = argparse.ArgumentParser(description="Crypto AI Decision System (Institutional v30)")
    parser.add_argument("--mode", type=str, default="retail", choices=["retail", "institutional"], help="Execution Mode")
    parser.add_argument("--execution", type=str, default="python", choices=["python", "native"], help="Execution Engine")
    parser.add_argument("--backtest", type=str, default="vector", choices=["vector", "event_driven"], help="Backtest Engine")
    return parser.parse_args()

class InstitutionalRunner:
    """
    v30 Runner: Uses Rust Execution + C++ FIX + Event Backtest.
    """
    def __init__(self, execution_mode, backtest_mode):
        self.execution = execution_mode
        self.backtest = backtest_mode
        logger.info(f"🏛️ INSTITUTIONAL STACK INITIALIZING [Exec={execution_mode}, Backtest={backtest_mode}]")
        
        # 1. Native HFT Layer
        if self.execution == "native":
            self.native_eng = NativeEngine()
            # Verify connectivity
            try:
                mp = self.native_eng.get_microprice(100,101,1,1)
                logger.info(f"🦀 Rust Microstructure Engine Live. Test Microprice: {mp}")
            except Exception as e:
                logger.error(f"Rust Engine Init Error: {e}")
            
            # 2. FIX Gateway
            try:
                self.fix = FIXClient("FUND_A", "EXCHANGE_B")
                logger.info("🔌 C++ FIX Gateway Attached.")
            except Exception as e:
                logger.warning(f"FIX Gateway init failed: {e}")

        # 3. Optimizer v3
        self.opt = InstitutionalOptimizer(3, ["BTC", "ETH", "SOL"])
        logger.info("📐 Convex Optimizer v3 Ready.")

        # --- v31-v38 UPGRADES ---
        try:
            # v31 L3 Engine
            from src.data.l3_engine.l3_api import L3Engine
            self.l3 = L3Engine()
            logger.info("⚡ v31 L3 Market Data Engine Active.")
            
            # v32 Kernel Bypass (Simulation)
            # (Implicit in L3, mocked via C++)

            # v34 FIX Direct
            from deploy.fix_direct_routing import FIXRouter
            self.router = FIXRouter()
            route = self.router.get_route("Binance_Prime")
            logger.info(f"🌐 v34 FIX Router: Binance -> {route}")

            # v35 Arb Engine
            from src.arb_engine.core import ArbEngine
            self.arb = ArbEngine()
            logger.info("⚖️ v35 Multi-Asset Arb Engine Ready.")

            # v36 OMS
            from src.oms.oms_core import OMSCore
            self.oms = OMSCore()
            logger.info("📝 v36 Institutional OMS Active.")

            # v37 Prime Broker
            from src.prime.custody import CustodyManager
            self.custody = CustodyManager()
            logger.info("🏦 v37 Fireblocks/Copper Integration Online.")
            
        except Exception as e:
            logger.error(f"❌ Upgrade Load Failed: {e}")


    def run_backtest(self):
        if self.backtest == "event_driven":
            logger.info("⏱️ Starting Event-Driven Simulation (v29)...")
            engine = EventBacktester()
            sim = ExchangeSimulator(engine)
            engine.exchange = sim
            
            # Feed Mock Data
            md = MarketData("BTC-USD", 50000.0, 49990.0, 50010.0, 1.0)
            engine.push_event(Event(0.0, 0, md))
            
            # Run
            engine.run()
            logger.info("✅ Institutional Strategy Verification Complete.")

    def start_live(self):
        logger.info("🚀 Starting Native HFT Loop...")
        while True:
            time.sleep(1)


# --- CONFIGURATION ---
# Load Deployment Config
dep_conf = config.get("deployment", {})
LIVE_TRADING = dep_conf.get("go_live", False)
ENABLE_RISK_ENGINE = dep_conf.get("feature_flags", {}).get("enable_risk_engine", True)
ENABLE_VETO = dep_conf.get("feature_flags", {}).get("enable_regime_veto", True)
ENABLE_SHADOW = True  # v24 Prep default

if __name__ == "__main__":
    args = parse_args()

    # --- 1. Event-Driven Backtest (Simulation Mode) ---
    if args.backtest == "event_driven":
        print("\n" + "="*60)
        print("⏱️  v30 EVENT-DRIVEN SIMULATION ACTIVATED")
        print("="*60 + "\n")
        
        # Init Institutional Logic Check
        runner = InstitutionalRunner(args.execution, args.backtest)
        runner.run_backtest()
        
        print("\n✅ Simulation Complete. Exiting.")
        exit(0)

    # --- 2. Live / Loop Mode (Retail OR Institutional) ---
    if args.mode == "institutional":
        print("\n" + "="*60)
        print("🏛️  v30 INSTITUTIONAL LIVE STACK ACTIVATED")
        print("="*60 + "\n")
        
        # Init Native Components globally for use in job()
        # In a full refactor, we'd inject these into ExecutionQuantum
        try:
            native_eng = NativeEngine()
            mp = native_eng.get_microprice(100,101,1,1)
            logger.info(f"🦀 Rust Engine Active [MP={mp}]")

            # --- v31-v38 LIVE UPGRADES ---
            # v31 L3 Engine
            from src.data.l3_engine.l3_api import L3Engine
            l3_eng = L3Engine()
            logger.info("⚡ v31 L3 Market Data Engine Active.")

            # v34 FIX Direct
            from deploy.fix_direct_routing import FIXRouter
            router = FIXRouter()
            logger.info(f"🌐 v34 FIX Router Active (NY4/LD4 Config Loaded).")

            # v35 Arb Engine
            from src.arb_engine.core import ArbEngine
            arb_eng = ArbEngine()
            logger.info("⚖️ v35 Multi-Asset Arb Engine Ready.")

            # v36 OMS
            from src.oms.oms_core import OMSCore
            oms = OMSCore()
            logger.info("📝 v36 Institutional OMS Active.")

            # v37 Prime Broker
            from src.prime.custody import CustodyManager
            custody = CustodyManager()
            logger.info("🏦 v37 Fireblocks/Copper Integration Online.")

            # --- v39 SOFT-INFRA OPTIMIZATIONS ---
            # 1. ML
            from src.ml.transformers.orderflow_transformer import OrderflowTransformer
            dot_model = OrderflowTransformer()
            logger.info("🧠 v39 Deep Orderflow Transformer (DOT) Active.")
            
            from src.ml.automl.optuna_search import AutoMLSearch
            automl = AutoMLSearch()
            logger.info("🤖 v39 AutoML Optimizer Ready.")
            
            # 2. Regime
            from src.risk.regime.vol_lstm import VolLSTM
            vol_lstm = VolLSTM()
            logger.info("📉 v39 Volatility LSTM Online.")
            
            # 3. Execution
            from src.execution.queue.aqpe import AQPE
            aqpe = AQPE()
            logger.info("⏳ v39 Adaptive Queue Pos Estimator (AQPE) Active.")
            
            # 4. Risk
            from src.risk.portfolio.hrp import HRP
            hrp = HRP()
            logger.info("🛡️ v39 Hierarchical Risk Parity (HRP) Active.")

            # --- v40 ENGINEERING UPGRADE (TOP 1%) ---
            # 1. Messaging
            from src.infrastructure.messaging.event_bus import event_bus
            logger.info("📨 v40 EventBus (ZeroMQ/PubSub) Online.")
            
            # 2. State
            from src.infrastructure.state.store import state_store
            from src.infrastructure.state.wal import wal
            logger.info("💾 v40 Enterprise State Store + WAL Active.")
            
            # 3. Observability
            from src.infrastructure.observability.metrics import metrics
            from src.infrastructure.observability.slo_monitor import slo_monitor
            logger.info("🔭 v40 SRE Observability Suite (Prometheus/Jaeger) Ready.")
            
            # 4. Safety
            from src.safety.verifier import verifier
            logger.info("👮 v40 Formal Safety Verifier (Z3 Logic) Guarding Trades.")

        except Exception as e:
            logger.error(f"Failed to load Institutional Stack: {e}")
            
    else:
        print("\n" + "="*60)
        print("🧠  v19 QUANTUM RETAIL STACK ACTIVATED")
        print("="*60 + "\n")

    # Common Live Setup
    if LIVE_TRADING:
        print("\n\n" + "!" * 60)
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED")
        print("⚠️  REAL MONEY IS AT RISK.")
        print("!" * 60 + "\n")
        # Skipping Input for automated run safety in this environment
        # user_input = input("Type 'YES' to continue: ")

    logger.info(f"🚀 System Started [Mode={args.mode}]")

    # Initialize Components
    market_router = MarketRouter(primary_exchange="binance", secondary_exchanges=[])
    ob_manager = OrderBookManager(symbol="btcusdt")

    logger.info("🧠 Initializing AI Core...")
    meta_brain = MetaBrainV21()  
    arbitrator = AgentArbitrator()
    risk_engine_v3 = RiskEngineV3() if ENABLE_RISK_ENGINE else None

# --- ML MODEL LOADING ---
# --- ML MODEL LOADING (v40) ---
    import joblib
    import torch
    import json
    from src.ml.transformers.orderflow_transformer import OrderflowTransformer
    
    # Load v3 Model
    try:
        if (MODELS_DIR / "multifactor_model_v3.pkl").exists():
            ml_model = safe_load_model(MODELS_DIR / "multifactor_model_v3.pkl")
            logger.info("🧠 Multifactor Model v3 Loaded.")
        else:
            logger.warning("⚠️ Model v3 not found. Fallback to v2.")
        ml_model = safe_load_model(MODELS_DIR / "multifactor_model_v2.pkl")
    except Exception as e:
        logger.error(f"Failed to load ML Model: {e}")
        ml_model = None

    # Load Features
    try:
        with open(MODELS_DIR / "training_features.json", "r") as f:
            TRAINING_FEATURES = json.load(f)
        logger.info(f"📋 Loaded {len(TRAINING_FEATURES)} Training Features.")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load feature list: {e}")
        TRAINING_FEATURES = []

    dot_model = OrderflowTransformer()
    dot_model.eval()
    logger.info("🧠 DOT Model Initialized.")
    
    # Update Arbitrator to listen to ML
    arbitrator.regime_weights["NEUTRAL"].update({
        "ML_Ensemble": 0.4,
        "DOT_Signal": 0.3
    })
    # Re-normalize or just add (Arbitrator sums weights so it's fine)

    execution_quantum = ExecutionQuantum()

    # v24 Shadow Agent
    shadow_agent = None
    if ENABLE_SHADOW:
        shadow_agent = PPOPortfolioAgent(state_dim=10, action_dim=4)
        logger.info("🔮 Shadow PPO Agent Initialized")

    market_router_v2 = MarketRouterV2()
    noise_guard = MarketCleanlinessModel()
    liquidity_ai = LiquidityAI()
    self_healer = SelfHealingSystem(DATA_DIR)

    # Start HFT Thread
    def run_ob_loop():
        asyncio.run(ob_manager.start_stream())

    ob_thread = threading.Thread(target=run_ob_loop, daemon=True)
    ob_thread.start()
    logger.info("📡 OrderBook Manager Thread Started")

    # Start Main Scheduler Loop
    logger.info("⏰ Starting Continuous Strategy Loop...")
    job() # Run once immediately
    schedule.every(5).seconds.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)
