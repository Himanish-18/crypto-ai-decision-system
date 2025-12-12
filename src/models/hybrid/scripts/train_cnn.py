import logging
from pathlib import Path

import pandas as pd

from src.models.hybrid.tiny_cnn import TinyCNNProxy

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainCNN")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "hybrid"
FEATURES_FILE = DATA_DIR / "features" / "features_1H_advanced.parquet"


def train():
    logger.info("🧠 Starting Tiny-CNN Training...")
    if not FEATURES_FILE.exists():
        logger.error("Features file not found")
        return

    df = pd.read_parquet(FEATURES_FILE).dropna()
    df["target"] = (df["btc_close"].shift(-1) > df["btc_close"]).astype(int)
    df = df.dropna()

    cnn = TinyCNNProxy()
    cnn.fit(df, target_col="target")
    cnn.save(MODELS_DIR / "tiny_cnn_weights.pth")


if __name__ == "__main__":
    train()
