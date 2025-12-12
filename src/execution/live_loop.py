import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import schedule

from src.execution.binance_executor import BinanceExecutor
from src.execution.executor_queue import ExecutorQueue
from src.execution.live_signal_engine import LiveSignalEngine
from src.execution.trading_decision import TradingDecision
from src.features.build_features import (add_lagged_features,
                                         add_rolling_features,
                                         add_ta_indicators, engineer_sentiment)
from src.guardian.safety_daemon import SafetyDaemon
from src.ingest.live_market_data import LiveMarketData
from src.monitor.data_drift import DriftDetector
from src.risk_engine.risk_module import RiskEngine
from src.utils.alerting import TelegramAlertHandler

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler(),
        TelegramAlertHandler(),  # Add Telegram Handler
    ],
)
logger = logging.getLogger("live_loop")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "model_xgb_v1.pkl"
SCALER_PATH = MODELS_DIR / "scaler_v1.pkl"
LOG_DIR = DATA_DIR / "execution" / "logs"


class LiveBotService:
    def __init__(self):
        self.is_running = False
        self.market_data_btc = None
        self.market_data_eth = None
        self.signal_engine = None
        self.risk_engine = None
        self.decision_engine = None
        self.executor = None
        self.executor_queue = None
        self.guardian = None
        self.last_trade_id = None  # Track processed trades

    def initialize(self):
        """Initialize all components."""
        logger.info("🔧 Initializing Bot Components...")

        self.market_data_btc = LiveMarketData(symbol="BTC/USDT", timeframe="1h")
        self.market_data_eth = LiveMarketData(symbol="ETH/USDT", timeframe="1h")

        self.signal_engine = LiveSignalEngine(MODEL_PATH, SCALER_PATH)
        self.risk_engine = RiskEngine()
        self.decision_engine = TradingDecision(self.risk_engine, LOG_DIR)

        self.executor = BinanceExecutor(testnet=True)  # Default to Testnet
        self.executor_queue = ExecutorQueue(self.executor)
        self.executor_queue.start()

        self.guardian = SafetyDaemon(DATA_DIR, initial_capital=10000.0)

        # Initialize Drift Detector
        self.drift_detector = DriftDetector()
        try:
            self.drift_detector.load_reference()
        except Exception as e:
            logger.error(f"Failed to load drift reference: {e}")

        # Initialize last trade ID
        recent_trades = self.executor.get_recent_trades(limit=1)
        if recent_trades:
            self.last_trade_id = recent_trades[-1]["id"]

        logger.info("✅ Initialization Complete.")

    def log_paper_trade(self, data: dict):
        """Log paper trading data to JSONL."""
        log_file = DATA_DIR / "execution" / "paper_trades.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp if missing
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat()

        with open(log_file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features on live data."""
        if "sentiment_mean" not in df.columns:
            df["sentiment_mean"] = 0.0
            df["sentiment_count"] = 0.0

        df = add_ta_indicators(df)
        df = add_rolling_features(df)
        df = add_lagged_features(df)
        df = engineer_sentiment(df)
        return df

    def job(self):
        """Main trading logic."""
        logger.info("⏰ Starting Trading Cycle...")

        try:
            # 1. Fetch Data
            df_btc = self.market_data_btc.fetch_candles(limit=1000)
            df_eth = self.market_data_eth.fetch_candles(limit=1000)

            if df_btc is None or df_eth is None:
                logger.warning("Insufficient data. Skipping cycle.")
                return

            df = pd.merge(df_btc, df_eth, on="timestamp", how="inner")

            if len(df) < 100:
                logger.warning("Insufficient merged data. Skipping cycle.")
                return

            # 2. Calculate Features
            df = self.calculate_features(df)

            # --- GUARDIAN CHECK 1: System Health ---
            if not self.guardian.check_system_health(
                self.signal_engine.model, self.signal_engine.scaler, df
            ):
                logger.critical(
                    "🛑 Guardian: System Health Check Failed. Aborting Cycle."
                )
                return

            # --- DRIFT CHECK (Daily) ---
            # Check drift if it's 00:00 UTC (approx) or just check every cycle if fast enough
            # For robustness, let's check if hour == 0
            current_hour = pd.Timestamp.utcnow().hour
            if current_hour == 0 and self.drift_detector.reference_data is not None:
                logger.info("🔍 Running Daily Drift Check...")
                # Use last 24h of data (approx 24 rows)
                recent_data = df.iloc[-24:]
                drift_report = self.drift_detector.check_drift(recent_data)

                if not drift_report.empty:
                    critical_drift = drift_report[drift_report["status"] == "CRITICAL"]
                    if not critical_drift.empty:
                        logger.critical(
                            f"🚨 CRITICAL DATA DRIFT DETECTED in {len(critical_drift)} features. PAUSING TRADING."
                        )
                        self.guardian.state["is_locked"] = True
                        self.guardian.state["lock_reason"] = (
                            "Critical Data Drift Detected"
                        )
                        self.guardian._save_state()
                        return

            # 3. Get Latest Closed Candle
            latest_candle = df.iloc[[-2]].copy().reset_index(drop=True)
            current_price = latest_candle["btc_close"].iloc[0]
            timestamp = latest_candle["timestamp"].iloc[0]

            logger.info(f"Processing Candle: {timestamp} | Price: {current_price}")

            # --- GUARDIAN CHECK 2: Financial Health ---
            current_balance = self.executor.get_balance("USDT")

            if current_balance is None:
                logger.warning(
                    "⚠️ Failed to fetch balance. Skipping cycle to avoid false alarms."
                )
                return

            # 2a. Check PnL & Kill Switch
            recent_trades = self.executor.get_recent_trades(limit=5)
            for trade in recent_trades:
                tid = trade["id"]
                # Process only new trades
                if self.last_trade_id is None or tid > self.last_trade_id:
                    # Calculate PnL (realizedPnl if available, else estimate)
                    # ccxt unified: 'info' usually has 'realizedPnl' for futures
                    pnl = float(trade.get("info", {}).get("realizedPnl", 0.0))

                    # If pnl is 0, it might be an open trade or data missing.
                    # Only count if it's a realized loss/gain (e.g. reduceOnly or close)
                    # For simplicity, if pnl != 0, we track it.
                    if pnl != 0:
                        logger.info(f"💰 Trade {tid} PnL: {pnl}")
                        self.guardian.check_trade_streak(pnl)

                    self.last_trade_id = tid

            if not self.guardian.check_financial_health(current_balance):
                logger.critical(
                    "🛑 Guardian: Financial Health Check Failed. Aborting Cycle."
                )
                return

            # --- GO-LIVE CHECKLIST: Balance Check ---
            # Abort if balance is dangerously low for live trading
            if not self.executor.testnet and current_balance < 100:
                logger.critical(
                    f"🛑 Go-Live Safety: Balance ${current_balance} < $100. Aborting."
                )
                return

            # --- GUARDIAN CHECK 3: Exposure ---
            current_exposure = self.executor.get_position_value()
            if not self.guardian.check_exposure(current_exposure, current_balance):
                logger.warning(
                    f"🛑 Guardian: Exposure Limit Reached (Exp: {current_exposure}). Skipping Trade."
                )
                return

            # 4. Generate Signal
            signal_output = self.signal_engine.process_candle(latest_candle)

            # --- GUARDIAN CHECK 4: Market Conditions ---
            if not self.guardian.check_market_conditions(
                signal_output["strategy_context"]
            ):
                logger.warning(
                    "🛑 Guardian: Market Condition Check Failed. Skipping Trade."
                )
                return

            # 5. Make Decision
            decision = self.decision_engine.make_decision(signal_output, current_price)

            # 6. Execute Order via Queue
            if decision["action"] == "BUY":
                logger.info(f"Available Balance: {current_balance} USDT")

                if current_balance > 10:
                    # --- GUARDIAN CHECK 5: Execution Safety ---
                    if not self.guardian.check_execution_safety(
                        current_price, current_price
                    ):
                        logger.warning(
                            "🛑 Guardian: Execution Safety Check Failed. Holding."
                        )
                        return

                    # Submit to Queue
                    order_params = {
                        "symbol": "BTC/USDT",
                        "side": "buy",
                        "amount": decision["size"],
                        "order_type": "market",
                    }
                    self.executor_queue.submit_order(order_params)

                    # Log Stop Loss
                    sl_price = decision["stops"]["stop_loss"]
                    logger.info(f"🛑 Stop Loss should be at: {sl_price}")

                else:
                    logger.warning("Insufficient USDT balance.")

            elif decision["action"] == "SELL":
                # Logic for selling would go here
                pass

            # --- LOG PAPER TRADE ---
            paper_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "price": current_price,
                "regime": signal_output.get("regime", "Unknown"),
                "signal_prob": float(signal_output.get("probability", 0.0)),
                "action": decision["action"],
                "size": decision.get("size", 0.0),
                "stop_loss": decision.get("stops", {}).get("stop_loss", 0.0),
                "take_profit": decision.get("stops", {}).get("take_profit", 0.0),
                "exposure": current_exposure,
                "equity": current_balance,
                "losing_streak": self.guardian.state["losing_streak"],
                "is_locked": self.guardian.state["is_locked"],
            }
            self.log_paper_trade(paper_log)
            logger.info(
                f"📝 Paper Trade Logged: {decision['action']} | Size: {decision.get('size', 0.0)}"
            )

            # 7. Check Queue Errors
            queue_errors = self.executor_queue.get_recent_errors()
            if queue_errors:
                for err in queue_errors:
                    logger.critical(f"🚨 Executor Queue Error: {err}")
                # Optional: Pause trading or notify guardian?
                # For now, just logging as critical is enough for the dashboard/logs.

            logger.info("💤 Cycle Complete. Waiting for next schedule.")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)

    def run(self):
        """Run the service loop."""
        self.initialize()
        self.is_running = True

        logger.info("🚀 Live Bot Service Started")

        # Schedule
        schedule.every().hour.at(":02").do(self.job)

        # Run once immediately
        self.job()

        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping Service...")
                self.stop()
            except Exception as e:
                logger.critical(f"🔥 Critical Service Error: {e}", exc_info=True)
                time.sleep(10)  # Wait before restart attempt

    def stop(self):
        self.is_running = False
        if self.executor_queue:
            self.executor_queue.stop()


if __name__ == "__main__":
    service = LiveBotService()
    service.run()
