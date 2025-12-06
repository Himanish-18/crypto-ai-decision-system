import pandas as pd
import logging
from pathlib import Path
from src.models.hybrid.tcn_lite import TCNLiteProxy

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainTCN")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "hybrid"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"

def train():
    logger.info("🌊 Starting TCN-Lite Training...")
    df = pd.read_parquet(FEATURES_FILE).dropna()
    df["target"] = (df["btc_close"].shift(-1) > df["btc_close"]).astype(int)
    df = df.dropna()
    
    tcn = TCNLiteProxy()
    tcn.fit(df, target_col="target")
    tcn.save(MODELS_DIR / "tcn_lite_weights.pth")

if __name__ == "__main__":
    train()
