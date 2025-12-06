import logging
import time
import pandas as pd
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
MODEL_PATH = MODELS_DIR / "multifactor_model.pkl" # Updated to new Regime Model
SCALER_PATH = MODELS_DIR / "scaler.pkl" # Updated/Dummy scaler path
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

        # Fetch ETH (Optional/Future use) - Commented out for speed unless needed
        # df_eth = asyncio.run(market_router.fetch_unified_candles("ETH/USDT", timeframe="1m", limit=1000))
        
        # Use BTC df
        df = df_btc
        
        if len(df) < 100:
             logger.warning("Insufficient data. Skipping cycle.")
             return

        # 2. Calculate Basic Features
        df = calculate_features(df)
        
        # NOTE: Advanced Features (Alphas, OrderFlow) are now handled internally by LiveSignalEngine
        # to ensure consistency with Training Loop.
            
        # --- GUARDIAN CHECK 1: System Health ---
            
        # --- GUARDIAN CHECK 1: System Health ---
        if not guardian.check_system_health(signal_engine.model, signal_engine.scaler, df):
            logger.critical("🛑 Guardian: System Health Check Failed. Aborting Cycle.")
            return
        
        # 3. Get Latest Closed Candle
        latest_candle = df.iloc[[-2]].copy().reset_index(drop=True)
        current_price = latest_candle["btc_close"].iloc[0]
        timestamp = latest_candle["timestamp"].iloc[0]
        
        logger.info(f"Processing Candle: {timestamp} | Price: {current_price}")
        
        # --- GUARDIAN CHECK 2: Financial Health ---
        current_balance = base_executor.get_balance("USDT")
        if not guardian.check_financial_health(current_balance):
            logger.critical("🛑 Guardian: Financial Health Check Failed. Aborting Cycle.")
            return
            
        # 4. Generate Signal
        signal_output = signal_engine.process_candle(latest_candle)
        
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

if __name__ == "__main__":
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
