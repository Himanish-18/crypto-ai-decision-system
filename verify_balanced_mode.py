from src.execution.live_signal_engine import LiveSignalEngine
from pathlib import Path

# Mock paths
MODEL_PATH = Path("data/models/multifactor_model.pkl")
SCALER_PATH = Path("data/models/scaler.pkl")

try:
    engine = LiveSignalEngine(MODEL_PATH, SCALER_PATH, balanced_mode=True)
    print(f"✅ LiveSignalEngine Instantiated. Balanced Mode: {engine.balanced_mode}")
except Exception as e:
    print(f"❌ Failed: {e}")
