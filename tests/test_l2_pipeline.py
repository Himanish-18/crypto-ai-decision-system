import pytest
import asyncio
from src.features.microstructure_features import MicrostructureFeatures
from src.models.live_inference import LiveInferenceEngine
from src.execution.micro_execution import MicroExecution
from unittest.mock import MagicMock, AsyncMock

@pytest.mark.asyncio
async def test_feature_computation():
    engine = MicrostructureFeatures()
    snapshot = {
        "timestamp": 1000,
        "order_book": {
            "bids": [[100, 10]],
            "asks": [[101, 10]]
        },
        "recent_trades": []
    }
    features = engine.compute_features(snapshot)
    assert features["mid_price"] == 100.5
    assert features["spread"] == 1.0

@pytest.mark.asyncio
async def test_inference_engine():
    # Mock models to avoid loading files
    engine = LiveInferenceEngine()
    engine.regime_model = MagicMock()
    engine.ensemble_model = MagicMock()
    engine.ensemble_model.predict_proba.return_value = [0.8] # High buy prob
    
    features = {"ofi": 10, "cvd_10s": 5, "mid_price": 100, "microprice": 100.1}
    signal = engine.predict(features)
    
    assert signal["direction"] == 1
    assert signal["confidence"] > 0.5

@pytest.mark.asyncio
async def test_execution_logic():
    exec_engine = MicroExecution(log_file="test_log.jsonl")
    exec_engine.log_trade = AsyncMock()
    
    signal = {"direction": 1, "confidence": 0.9}
    snapshot = {
        "symbol": "BTCUSDT",
        "order_book": {
            "bids": [[100, 10]],
            "asks": [[100.05, 10]] # Tight spread
        }
    }
    
    await exec_engine.execute(signal, snapshot)
    exec_engine.log_trade.assert_called_once()
    
    # Test Spread Filter
    snapshot_wide = {
        "symbol": "BTCUSDT",
        "order_book": {
            "bids": [[100, 10]],
            "asks": [[105, 10]] # Wide spread
        }
    }
    exec_engine.log_trade.reset_mock()
    await exec_engine.execute(signal, snapshot_wide)
    exec_engine.log_trade.assert_not_called()

if __name__ == "__main__":
    # Manual run if pytest not available
    asyncio.run(test_feature_computation())
    asyncio.run(test_inference_engine())
    asyncio.run(test_execution_logic())
    print("All tests passed!")
