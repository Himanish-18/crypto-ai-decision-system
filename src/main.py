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

def job():
    logger.info("⏰ Starting Trading Cycle...")
    
    try:
        # 1. Fetch Data (History for features)
        # Using v8 MarketRouter (Unified 1m Candles)
        
        # Fetch BTC
        df_btc = asyncio.run(market_router.fetch_unified_candles("BTC/USDT", timeframe="1m", limit=1000))
        if df_btc is None or len(df_btc) < 100:
            logger.warning("Insufficient BTC data (MarketRouter). Skipping cycle.")
            return
        df_btc = df_btc.rename(columns={"volume": "btc_volume"})
        df_btc["volume"] = df_btc["btc_volume"] # Legacy support
        df_btc["high"] = df_btc["btc_high"]
        df_btc["low"] = df_btc["btc_low"]
        df_btc["close"] = df_btc["btc_close"]
        df_btc["open"] = df_btc["btc_open"]

        # Fetch ETH (Required for TCN Model)
        df_eth = asyncio.run(market_router.fetch_unified_candles("ETH/USDT", timeframe="1m", limit=1000))
        if df_eth is not None:
             df_eth = df_eth.rename(columns={"volume": "eth_volume"})
             
             # Merge
             df = pd.merge(df_btc, df_eth, on="timestamp", how="inner", suffixes=("", "_eth"))
        else:
             logger.warning("Insufficient ETH data. Model might fail if features missing.")
             df = df_btc
             if "btc_volume" not in df.columns: df = df.rename(columns={"volume": "btc_volume"})

        if len(df) < 100:
             logger.warning("Insufficient data after merge. Skipping cycle.")
             return

        # 2. Calculate Basic Features
        df = calculate_features(df)
            
        # --- GUARDIAN CHECK 1: System Health ---
        if not guardian.check_system_health(signal_engine.model, signal_engine.scaler, df):
            logger.critical("🛑 Guardian: System Health Check Failed. Aborting Cycle.")
            return
        
        # 3. Get Latest Context (Pass History Window)
        # Pass last 300 candles (excluding the very last if likely open/incomplete, but fetch_unified returns closed usually? 
        # MarketRouter usually returns completed candles. Latest might be open depending on exchange.
        # Safest is to take up to -1 if we are sure -1 is open.
        # Assuming fetch returns open candle at end -> slice up to -1.
        # If fetch returns only closed -> slice all.
        # Let's assume -1 is open/latest.
        
        # Define window size for models (some need 64, some 100)
        window_size = 300
        latest_window = df.iloc[-(window_size+1):-1].copy().reset_index(drop=True)
         
        current_price = latest_window["btc_close"].iloc[-1]
        timestamp = latest_window["timestamp"].iloc[-1]
        
        logger.info(f"Processing Candle: {timestamp} | Price: {current_price}")
        
        # --- GUARDIAN CHECK 2: Financial Health ---
        current_balance = base_executor.get_balance("USDT")
        if not guardian.check_financial_health(current_balance):
            logger.critical("🛑 Guardian: Financial Health Check Failed. Aborting Cycle.")
            return
            
        # 4. Generate Signal
        signal_output = signal_engine.process_candle(latest_window)
        
        # --- GUARDIAN CHECK 3: Market Conditions ---
        if not guardian.check_market_conditions(signal_output["strategy_context"]):
            logger.warning("🛑 Guardian: Market Condition Check Failed. Skipping Trade.")
            return
        
        # 5. Make Decision
        decision = decision_engine.make_decision(signal_output, current_price)
        
        # 6. Execute Order
        if decision["action"] == "BUY":
            logger.info(f"Available Balance: {current_balance} USDT")
            
            if current_balance > 10: # Min balance check
                # --- GUARDIAN CHECK 4: Execution Safety ---
                # Assuming market order, so order price ~ current price
                if not guardian.check_execution_safety(current_price, current_price):
                    logger.warning("🛑 Guardian: Execution Safety Check Failed. Holding.")
                    return
                    
                # Get Microstructure Metrics
                ob_metrics = ob_manager.get_latest_metrics()
                
                # Execute Buy (Smart)
                # Ensure we run async method in sync context
                try:
                    order = asyncio.run(smart_executor.execute_order(
                        symbol="BTC/USDT",
                        side="buy",
                        amount=decision["size"],
                        style="AUTO",
                        microstructure=ob_metrics,
                        stops=decision["stops"]
                    ))
                except Exception as ex_err:
                    logger.error(f"Execution Error: {ex_err}")
                    order = None
                
                if order:
                    logger.info(f"✅ BUY Order Executed: {order}")
                    sl_price = decision["stops"]["stop_loss"]
                    logger.info(f"🛑 Stop Loss should be at: {sl_price}")
            else:
                logger.warning("Insufficient USDT balance.")
                
        elif decision["action"] == "SELL":
            # Logic for selling/closing position would go here
            pass
            
        logger.info("💤 Cycle Complete. Waiting for next schedule.")
        
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}", exc_info=True)


# --- CONFIGURATION ---
LIVE_TRADING = True # Set to True to enable Real-Money Safety Checks & Execution

if __name__ == "__main__":
    # Safety Confirmation
    if LIVE_TRADING:
        print("\n\n" + "!"*60)
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED")
        print("⚠️  REAL MONEY IS AT RISK.")
        print("⚠️  ENSURE YOU HAVE SET 'BINANCE_TESTNET=True' IN .env FOR TESTING.")
        print("!"*60 + "\n")
        
        # Determine if we are really on mainnet?
        # We can't know for sure until Executor inits, but we can warn generally.
        
        user_input = input("Type 'YES' to continue with LIVE TRADING: ")
        if user_input.strip() != "YES":
            print("❌ Start Aborted.")
            exit(0)
            
    logger.info("🚀 Live Trading Bot Started")
    
    # Initialize Components
    # v8 Market Router
    market_router = MarketRouter(primary_exchange="binance", secondary_exchanges=["bybit", "okx"])
    logger.info("✅ MarketRouter Initialized")
    
    signal_engine = LiveSignalEngine(MODEL_PATH, SCALER_PATH, balanced_mode=True)
    risk_engine = RiskEngine() 
    decision_engine = TradingDecision(risk_engine, LOG_DIR)
    
    # Execution Stack
    base_executor = BinanceExecutor() # Configured via env vars (Defaults to Dry Run if missing)
    smart_executor = SmartExecutor(base_executor)
    
    # Order Book Manager (Async in Thread)
    ob_manager = OrderBookManager(symbol="btcusdt")
    def run_ob_loop():
        asyncio.run(ob_manager.start_stream())
    
    ob_thread = threading.Thread(target=run_ob_loop, daemon=True)
    ob_thread.start()
    logger.info("📡 OrderBook Manager Thread Started")
    
    # Initialize Guardian
    guardian = SafetyDaemon(DATA_DIR, initial_capital=10000.0) # Set initial capital correctly!
    
    # Run once immediately to verify
    job()
    
    # Schedule every minute (since we are on 1m timeframe now)
    schedule.every().minute.at(":02").do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
