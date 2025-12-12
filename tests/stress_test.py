import logging
import statistics
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# --- MOCKS ---
# Removed global sys.modules hacking to prevent test pollution.
# We will use patch.dict context manager during execution or import.

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StressTest")

try:
    from src.execution.live_signal_engine import LiveSignalEngine
    from src.features.sentiment_features import SentimentFeatures
    from src.models.hybrid.tcn_lite import TCNLiteProxy
except ImportError:
    import os

    sys.path.append(os.getcwd())
    from src.execution.live_signal_engine import LiveSignalEngine
    from src.features.sentiment_features import SentimentFeatures
    from src.models.hybrid.tcn_lite import TCNLiteProxy


def generate_mock_candle(timestamp, price=50000.0, vol=100.0):
    """Generate a valid looking candle row."""
    return pd.DataFrame(
        {
            "timestamp": [timestamp],
            "open": [price],
            "high": [price * 1.01],
            "low": [price * 0.99],
            "close": [price],
            "volume": [vol],
            "btc_close": [price],
            "btc_volume": [vol],
            "btc_high": [price * 1.01],
            "btc_low": [price * 0.99],
            "fundingRate": [0.0001],
            "openInterest": [100000.0],
        }
    )


def setup_engine():
    """Initialize engine with mocks."""
    mock_model_path = MagicMock()
    mock_model_path.parent.parent = MagicMock()
    mock_model_path.exists.return_value = False
    mock_scaler_path = MagicMock()

    # Context manager for sys.modules patches
    patcher = patch.dict(sys.modules, {
        "src.data.deribit_vol_monitor": MagicMock(),
        "src.risk_engine.iv_guard": MagicMock(),
        "torch": MagicMock(),
        "src.models.portfolio_rl.ppo_agent": MagicMock()
    })
    patcher.start()

    with patch.object(LiveSignalEngine, "load_artifacts"), patch.object(
        LiveSignalEngine, "load_hybrid_models"
    ):
        engine = LiveSignalEngine(mock_model_path, mock_scaler_path)

        # Manually wire valid components
        engine.sentiment_gen = SentimentFeatures()
        engine.selected_features = None

        # Mock ML models to return fast constants (we test engine logic, not ML inference speed)
        # But we DO want to test feature engineering speed mostly.
        engine.tcn_lite = MagicMock()
        engine.tcn_lite.predict_trend.return_value = 0.5
        engine.tiny_cnn = MagicMock()
        engine.tiny_cnn.predict_score.return_value = 0.5

        engine.meta_regime = MagicMock()
        engine.meta_regime.predict.return_value = {
            "predicted_regime": "STABLE",
            "confidence": 0.5,
        }

        # Mocks for other logic
        from src.execution.hft_orderbook import HftOrderBook

        engine.hft_ob = HftOrderBook()
        engine.fill_prob_model = MagicMock()
        engine.fill_prob_model.estimate_fill_prob.return_value = 0.8
        engine.tcm = MagicMock()
        engine.tcm.estimate_cost.return_value = 0.0001
        engine.tcm.is_profitable.return_value = True
        engine.meta_safety = MagicMock()
        engine.meta_safety.check_safety.return_value = True
        engine.ppo_policy = MagicMock()
        engine.ppo_policy.get_action.return_value = 1.0
        engine.kalman = MagicMock()
        engine.kalman.smooth.return_value = 0.6
        engine.pem = MagicMock()
        engine.pem.predict.return_value = {"exit_signal": False, "panic_score": 0.0}
        engine.of_gen = MagicMock()
        engine.of_gen.calculate_features.side_effect = lambda x: x
        engine.vol_adaptive = MagicMock()
        engine.vol_adaptive.get_threshold.return_value = 0.5
        engine.trend_depth = MagicMock()
        engine.trend_depth.calculate.return_value = 0.5

        # Fix: Initialize attributes skipped by patched load_artifacts
        engine.v19_stacker = None
        engine.xgb_stacker = None
        engine.v19_failure_detector = None

        return engine


def run_latency_test(engine, n=1000):
    logger.info(f"🏎️  Starting Latency Test ({n} iterations)...")
    latencies = []

    start_time = pd.Timestamp.now(tz=None)  # Naive start

    for i in range(n):
        ts = start_time + pd.Timedelta(minutes=i)
        df = generate_mock_candle(ts)

        t0 = time.time()
        try:
            res = engine.process_candle(df)
        except Exception as e:
            logger.error(f"Crash during latency test at iter {i}: {e}")
            break
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)  # ms

    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18]
    p99 = statistics.quantiles(latencies, n=100)[98]
    max_lat = max(latencies)

    logger.info(f"📊 Latency Results:")
    logger.info(f"   Avg: {statistics.mean(latencies):.2f}ms")
    logger.info(f"   P50: {p50:.2f}ms")
    logger.info(f"   P95: {p95:.2f}ms")
    logger.info(f"   P99: {p99:.2f}ms")
    logger.info(f"   Max: {max_lat:.2f}ms")

    return latencies


def run_fuzz_test(engine, n=100):
    logger.info(f"🌪️  Starting Fuzz/Robustness Test ({n} iterations)...")
    crashes = 0
    start_time = pd.Timestamp.now(tz=None)

    for i in range(n):
        ts = start_time + pd.Timedelta(minutes=i)
        df = generate_mock_candle(ts)

        # Inject Chaos
        chaos_type = i % 5

        if chaos_type == 0:
            # Missing arbitrary columns
            if "volume" in df.columns:
                df = df.drop(columns=["volume"])
            if "btc_volume" in df.columns:
                df = df.drop(columns=["btc_volume"])
            desc = "Missing Volume"

        elif chaos_type == 1:
            # NaNs in critical fields
            df["btc_close"] = np.nan
            desc = "NaN Close"

        elif chaos_type == 2:
            # Infinity
            df["fundingRate"] = np.inf
            desc = "Inf Funding"

        elif chaos_type == 3:
            # Zero values
            df["btc_close"] = 0.0
            df["high"] = 0.0
            desc = "Zero Price"

        else:
            # Type Mismatch (String in float col) - Pandas usually handles or crashes hard.
            # We skip strict type fuzzing for now as pandas converts or errors early.
            # Let's try extremley large numbers
            df["btc_close"] = 1e18
            desc = "Huge Number"

        try:
            res = engine.process_candle(df)
            # Assert keys exist
            required = ["signal", "strategy_context", "prediction_prob"]
            if not all(k in res for k in required):
                logger.error(
                    f"❌ Malformed output for {desc}: Missing keys {set(required) - set(res.keys())}"
                )
                crashes += 1
        except Exception as e:
            logger.error(f"❌ CRASH for {desc}: {e}")
            crashes += 1

    logger.info(f"🛡️  Fuzz Test Complete. Crashes: {crashes}/{n}")
    return crashes


if __name__ == "__main__":
    eng = setup_engine()

    # 1. Warmup
    run_latency_test(eng, n=10)

    # 2. Main Latency
    lats = run_latency_test(eng, n=1000)

    # 3. Fuzzing
    crashes = run_fuzz_test(eng, n=500)

    if crashes == 0 and statistics.mean(lats) < 50:
        logger.info("✅ STRESS TEST PASSED")
        sys.exit(0)
    elif crashes > 0:
        logger.error("❌ STRESS TEST FAILED: Crashes detected.")
        sys.exit(1)
    else:
        logger.warning("⚠️ STRESS TEST PASSED WITH WARNINGS (High Latency).")
        sys.exit(0)
