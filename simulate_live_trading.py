import logging
import pandas as pd
from pathlib import Path
from src.execution.live_signal_engine import LiveSignalEngine
from src.risk_engine.risk_module import RiskEngine
from src.execution.trading_decision import TradingDecision

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation")

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "model_xgb_v1.pkl"
SCALER_PATH = MODELS_DIR / "scaler_v1.pkl"
LOG_DIR = DATA_DIR / "execution" / "logs"

def run_simulation():
    logger.info("🚀 Starting Live Trading Simulation...")
    
    # Initialize Components
    signal_engine = LiveSignalEngine(MODEL_PATH, SCALER_PATH)
    risk_engine = RiskEngine()
    decision_engine = TradingDecision(risk_engine, LOG_DIR)
    
    # Load Data (Simulating Stream)
    df = pd.read_parquet(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Use last 5000 candles for simulation to ensure we catch trades
    sim_data = df.iloc[-5000:].reset_index(drop=True)
    
    # Pre-calculate ATR pct for simulation (since we don't have expanding window in live engine yet)
    # In real live engine, we'd need to maintain history.
    sim_data["atr_pct"] = sim_data["btc_atr_14"].rank(pct=True) # Approximation for this snippet
    
    logger.info(f"Processing {len(sim_data)} candles...")
    
    buy_signals = 0
    
    for i in range(len(sim_data)):
        candle = sim_data.iloc[[i]]
        current_price = candle["btc_close"].iloc[0]
        
        # 1. Generate Signal
        signal_output = signal_engine.process_candle(candle)
        
        # 2. Make Decision
        decision = decision_engine.make_decision(signal_output, current_price)
        
        if decision["action"] == "BUY":
            buy_signals += 1
            logger.info(f"[{decision['timestamp']}] BUY SIGNAL! Size: {decision['size']:.4f} BTC")
        elif i < 10 or i % 500 == 0:
            logger.info(f"[{decision['timestamp']}] SKIP: {decision['reason']} | Prob: {signal_output['prediction_prob']:.4f} | RSI: {signal_output['strategy_context']['rsi']:.1f}")
            
    logger.info(f"Simulation Complete. Total Buy Signals: {buy_signals}")
    logger.info(f"Logs saved to {LOG_DIR}")

if __name__ == "__main__":
    run_simulation()
