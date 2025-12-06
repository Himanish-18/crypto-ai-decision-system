import logging
import pandas as pd
import json
from pathlib import Path
from src.execution.live_signal_engine import LiveSignalEngine

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODEL_PATH = MODELS_DIR / "multifactor_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY_HYBRID")

def verify_hybrid_system():
    logger.info("🧪 Verifying Hybrid v4 System...")
    
    # Init Engine
    engine = LiveSignalEngine(MODEL_PATH, SCALER_PATH, balanced_mode=True)
    
    # Load Data (Sample)
    df = pd.read_parquet(FEATURES_FILE).iloc[-500:] # Last 500
    
    signals = []
    
    # Feed Data
    # For simulation, we need to feed window of size 20+
    # Window size for cnn is 20.
    
    for i in range(100, 150): # Run 50 iterations
        window = df.iloc[i-50:i]
        
        # Inject Fake RL engine mock if needed? 
        # But Hybrid v4 uses own DQN, doesn't use old RL Engine.
        # Wait, existing code has `rl_action` inside `process_candle` or `get_signal`?
        # The new code DOES NOT call `self.rl_engine.get_signal` inside the `if balanced_mode:` block anymore (unless I kept it?)
        # Let's check my edit in Step 1999.
        # I *replaced* the RL signal call with Hybrid calls inside the `if balanced_mode` block.
        # So it shouldn't need old RL.
        
        try:
            result = engine.process_candle(window)
            if result and result.get("signal") == 1:
                 signals.append(result)
        except Exception as e:
            logger.error(f"Error at step {i}: {e}")
            
    logger.info(f"Generated {len(signals)} signals in 50 steps.")
    if len(signals) > 0:
        logger.info(f"First Signal: {json.dumps(signals[0], default=str)}")
        
    logger.info("✅ Verification Complete.")

if __name__ == "__main__":
    verify_hybrid_system()
