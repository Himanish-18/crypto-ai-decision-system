import logging
import pandas as pd
from pathlib import Path
from src.execution.live_signal_engine import LiveSignalEngine

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_live_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_mega_alpha.csv"

def main():
    logger.info("🚀 Starting Live Engine Test...")
    
    # Paths
    model_path = MODELS_DIR / "best_model_xgb_opt.pkl"
    scaler_path = MODELS_DIR / "scaler_opt.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        logger.error("❌ Model or Scaler not found. Run model_improvement.py first.")
        return

    # Initialize Engine
    try:
        engine = LiveSignalEngine(model_path=model_path, scaler_path=scaler_path)
        logger.info("✅ LiveSignalEngine initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")
        return

    # Load Sample Data
    if not FEATURES_FILE.exists():
        logger.error(f"❌ Features file not found: {FEATURES_FILE}")
        return
        
    df = pd.read_csv(FEATURES_FILE)
    # Take the last row as "current candle"
    latest_candle = df.tail(1)
    
    logger.info(f"📊 Processing candle from {latest_candle['timestamp'].values[0]}...")
    
    try:
        result = engine.process_candle(latest_candle)
        logger.info("✅ Signal Generated:")
        logger.info(result)
        
        if result['signal'] == 1:
            logger.info("🟢 BUY SIGNAL")
        else:
            logger.info("⚪ NO SIGNAL")
            
    except Exception as e:
        logger.error(f"❌ Failed to process candle: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
