import logging
from pathlib import Path
import pandas as pd
from src.models.regime_detector import RegimeDetector

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_regime")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_PATH = MODELS_DIR / "regime_model.pkl"

def main():
    logger.info("📥 Loading data...")
    df = pd.read_parquet(FEATURES_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Data Shape: {df.shape}")
    
    # Initialize and Train
    detector = RegimeDetector(n_components=4, n_iter=100)
    detector.fit(df)
    
    # Save
    logger.info(f"💾 Saving model to {OUTPUT_PATH}...")
    detector.save(OUTPUT_PATH)
    
    # Verify
    latest_regime = detector.predict(df.tail(100))
    logger.info(f"Latest Regime: {latest_regime}")

if __name__ == "__main__":
    main()
