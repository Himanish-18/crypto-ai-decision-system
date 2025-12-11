import logging
import time
import pandas as pd
import numpy as np
import schedule
from pathlib import Path
from datetime import datetime
import pytz

from src.execution.trading_decision import TradingDecision
from src.execution.binance_executor import BinanceExecutor
from src.execution.smart_executor import SmartExecutor
from src.features.orderbook_features import OrderBookManager
from src.guardian.safety_daemon import SafetyDaemon
from src.data.market_router import MarketRouter
from src.execution.live_signal_engine import LiveSignalEngine
from src.risk_engine.risk_module import RiskEngine
import threading
import asyncio
from src.features.build_features import add_ta_indicators, add_rolling_features, add_lagged_features, engineer_sentiment
from src.features.alpha_signals import AlphaSignals
from src.features.orderflow_features import OrderFlowFeatures
from src.portfolio.engine_v15 import PortfolioCoordinator

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "multifactor_model.pkl" 
SCALER_PATH = MODELS_DIR / "scaler.pkl" 
LOG_DIR = DATA_DIR / "execution" / "logs"

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features on live data."""
    # Mock Sentiment if missing (Live feed not implemented)
    if "sentiment_mean" not in df.columns:
        df["sentiment_mean"] = 0.0
        df["sentiment_count"] = 0.0
        
    # 1. Technical Indicators
    df = add_ta_indicators(df)
    
    # 2. Rolling Stats
    df = add_rolling_features(df)
    
    # 3. Lagged Features
    df = add_lagged_features(df)
    
    # 4. Sentiment (Upgraded)
    df = engineer_sentiment(df)

    # 5. Alpha Signals
    alpha_signals = AlphaSignals()
    df = alpha_signals.compute_all(df, symbol="btc")
    if "eth_close" in df.columns:
        df = alpha_signals.compute_all(df, symbol="eth")
    
    # 6. Order Flow Features
    of_feats = OrderFlowFeatures()
    df = of_feats.compute_all(df, symbol="btc")
    if "eth_close" in df.columns:
        df = of_feats.compute_all(df, symbol="eth")
    
    # Fill NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(0)
    
    return df

# --- v19 QUANTUM UPGRADE ---
from src.decision.meta_brain_titan import MetaBrainHardVeto
from src.decision.meta_brain_v21 import MetaBrainV21
from src.execution.execution_quantum import ExecutionQuantum
from src.decision.arbitrator import AgentArbitrator
from src.risk_engine.risk_v3 import RiskEngineV3
from src.market.router_v2 import MarketRouterV2
from src.ml.noise.cleanliness import MarketCleanlinessModel
from src.execution.liquidity_ai import LiquidityAI
from src.maintenance.self_heal import SelfHealingSystem

from src.rl.ppo_portfolio import PPOPortfolioAgent # v23
import yaml # Added for config loading

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
    formatter = logging.Formatter('%(asctime)s,%(message)s')
    handler.setFormatter(formatter)
    acc_logger.addHandler(handler)

# Global State for Validation
LAST_PRED_SCORE = None
LAST_PRED_PRICE = None
LAST_PRED_DIR = "Neutral"

def job():
    global LAST_PRED_SCORE, LAST_PRED_PRICE, LAST_PRED_DIR
    logger.info("⏰ Starting v19 QUANTUM Intelligence Cycle...")
    
    try:
        # 1. System Self-Check
        self_healer.monitor_performance(accuracy_window=0.9, latency_ms=5.0)
        
        # 2. Autonomous Market Switching
        focus_ticker = market_router_v2.scan_markets({}) 
        logger.info(f"🔭 QUANTUM Auto-Focus: {focus_ticker}")
        
        # 3. Data Fetch
        pair = focus_ticker
        df = asyncio.run(market_router.fetch_unified_candles(pair, timeframe="1m", limit=100))
        
        if df is None or df.empty:
            logger.warning(f"No Data for {focus_ticker}")
            return

        if "close" in df.columns:
            current_price = df["close"].iloc[-1]
        else:
            close_col = [c for c in df.columns if "close" in c][0]
            current_price = df[close_col].iloc[-1]
            df["close"] = df[close_col]
            if "open" not in df.columns: df["open"] = df[[c for c in df.columns if "open" in c][0]]
            if "high" not in df.columns: df["high"] = df[[c for c in df.columns if "high" in c][0]]
            if "low" not in df.columns: df["low"] = df[[c for c in df.columns if "low" in c][0]]
            if "volume" not in df.columns: df["volume"] = df[[c for c in df.columns if "volume" in c][0]]
            
        # --- RESTORED LOGIC START: Prediction Validation ---
        if LAST_PRED_SCORE is not None and LAST_PRED_PRICE is not None:
            ret_since = (current_price - LAST_PRED_PRICE) / LAST_PRED_PRICE
            real_dir = "Up" if ret_since > 0.0015 else "Down" if ret_since < -0.0015 else "Flat"
            
            is_correct = False
            if (LAST_PRED_DIR == "Bullish" and real_dir == "Up") or \
               (LAST_PRED_DIR == "Bearish" and real_dir == "Down"):
                is_correct = True
            elif real_dir == "Flat" and LAST_PRED_DIR in ["Bearish", "Neutral"]:
                 is_correct = True
            
            icon = "✅" if is_correct else "❌"
            logger.info(f"🎯 Accuracy Check: Pred [{LAST_PRED_DIR} {LAST_PRED_SCORE:.2f}] vs Real [{real_dir} {ret_since*100:.3f}%] -> {icon}")
            
            # Write to predictions.log
            acc_logger.info(f"{LAST_PRED_DIR},{LAST_PRED_SCORE:.4f},{real_dir},{ret_since:.6f},{icon}")
        # --- RESTORED LOGIC END ---
        
        # 4. Noise Immunity Check
        noise_score = noise_guard.analyze_cleanliness(df)
        if noise_score > 0.65:
            logger.warning(f"🌪 Market Choppy (Score {noise_score:.2f}). QUANTUM Blocking Trades.")
            return

        # 5. QUANTUM Intelligence Layer
        market_payload = {
            "symbol": focus_ticker,
            "candles": df,
            "microstructure": ob_manager.get_latest_metrics(),
            "funding_rate": 0.0001, 
            "volatility": 0.02 
        }
        
        # A. Meta-Brain (Hard Veto Only)
        # We use MetaBrain to check for Macro Veto. If it returns HOLD, we stop.
        # But for active signals, we use the Arbitrator.
        macro_decision = meta_brain.think(market_payload)
        if macro_decision["action"] == "HOLD" and macro_decision.get("agent") in ["TITAN_VETO", "TITAN_META_LABEL"]:
             logger.warning(f"🛡️ QUANTUM VETO: {macro_decision['reason']}")
             telemetry.inc("meta_brain_veto_count")
             return

        # B. Agent Arbitrator (The Voting Booth)
        # Stub: Simulating raw agent signals for arbitration
        # In full prod, we would call each agent.analyze() here.
        raw_signals = {
            "MomentumHunter": {"signal": 1.0, "confidence": 0.8},
            "MeanRevGhost": {"signal": -1.0, "confidence": 0.3},
            "VolOracle": {"signal": 0.0, "confidence": 0.0}
        }
        regime = market_payload.get("regime", "NEUTRAL") # Injected by MetaBrain or re-fetched
        
        arbitrated_decision = arbitrator.arbitrate(raw_signals, regime)
        
        # 6. Risk Engine v3 (Factor Check)
        if risk_engine_v3:
            market_factors = {"beta": 1.0, "liquidity": 1.0} # Stub
            is_safe = risk_engine_v3.check_trade_risk(arbitrated_decision, market_factors)
            
            if not is_safe:
                logger.warning("🛑 RISK VETO: Factor Exposure Limit Reached.")
                return
        else:
             # If Risk Engine disabled, assume safe
             pass
            
        # 7. Execution Quantum
        if arbitrated_decision["action"] != "HOLD":
            # Decide Intent
            intent = liquidity_ai.analyze_intent(arbitrated_decision["action"], arbitrated_decision["size"], market_payload)
            logger.info(f"🤖 QUANTUM Intent: {intent['type']}")
            
            # Execute with Microprice
            execution_quantum.execute_order(arbitrated_decision, market_payload)
            
        # 8. v24 Shadow Portfolio Agent
        if shadow_agent:
            # Construct Mock State (Returns, Vols, Weights)
            # Assuming 3 assets (BTC, ETH, SOL) + Cash as in Env
            # Real implementation would fetch portfolio vector
            mock_obs = np.random.normal(0, 1, 10) # 3*2 + 4 dim
            try:
                alloc, _ = shadow_agent.select_action(mock_obs)
                logger.info(f"🔮 SHADOW PORTFOLIO: Allocation = {alloc}")
            except Exception as e_shadow:
                logger.error(f"Shadow Agent Error: {e_shadow}")
            
        # --- RESTORED LOGIC START: Update State ---
        LAST_PRED_PRICE = current_price
        # Map Decision to Score
        # Decision has action=BUY/SELL/HOLD, size=0.0-1.0
        score = arbitrated_decision.get("size", 0.0)
        action = arbitrated_decision.get("action", "HOLD")
        
        if action == "BUY":
            LAST_PRED_DIR = "Bullish"
            LAST_PRED_SCORE = 0.5 + (score * 0.5) # Map 0-1 to 0.5-1.0
        elif action == "SELL":
            LAST_PRED_DIR = "Bearish"
            LAST_PRED_SCORE = 0.5 - (score * 0.5) # Map 0-1 to 0.5-0.0
        else:
            LAST_PRED_DIR = "Neutral"
            LAST_PRED_SCORE = 0.5
            
        # --- RESTORED LOGIC END ---

        logger.info("💤 Cycle Complete.")

    except Exception as e:
        logger.error(f"Error in v19 cycle: {e}", exc_info=True)
        self_healer.restart_hft_layer(ob_manager)


# --- CONFIGURATION ---
# Load Deployment Config
dep_conf = config.get("deployment", {})
LIVE_TRADING = dep_conf.get("go_live", False)
ENABLE_RISK_ENGINE = dep_conf.get("feature_flags", {}).get("enable_risk_engine", True)
ENABLE_VETO = dep_conf.get("feature_flags", {}).get("enable_regime_veto", True)
ENABLE_SHADOW = True # v24 Prep default

if __name__ == "__main__":
    if LIVE_TRADING:
        print("\n\n" + "!"*60)
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED (v19 QUANTUM)")
        print("⚠️  REAL MONEY IS AT RISK.")
        print("!"*60 + "\n")
        
        user_input = input("Type 'YES' to continue: ")
        if user_input.strip() != "YES":
            exit(0)
            
    logger.info(f"🚀 Live Trading Bot Started (v19 QUANTUM System) [LIVE={LIVE_TRADING}]")
    logger.info(f"🚩 Feature Flags: RiskEngine={ENABLE_RISK_ENGINE}, RegimeVeto={ENABLE_VETO}")
    
    # Initialize Core
    market_router = MarketRouter(primary_exchange="binance", secondary_exchanges=[])
    ob_manager = OrderBookManager(symbol="btcusdt")
    
    # Initialize v19 QUANTUM Stack
    logger.info("🧠 Initializing QUANTUM Core...")
    meta_brain = MetaBrainV21() # v21 Regime Detection (Veto Enabled by Class Logic)
    arbitrator = AgentArbitrator()
    risk_engine_v3 = RiskEngineV3() if ENABLE_RISK_ENGINE else None

    execution_quantum = ExecutionQuantum()
    
    # v24 Shadow Agent
    shadow_agent = None
    if ENABLE_SHADOW:
        # 3 Assets + Cash = 4 dim action, 10 dim state
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
    
    # Start Scheduler
    job()
    schedule.every().minute.at(":05").do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
