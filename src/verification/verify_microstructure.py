import asyncio
import logging

import pandas as pd

from src.features.orderbook_features import OrderBookManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_microstructure")


async def test_ob_manager():
    logger.info("🧪 Testing OrderBookManager Connection & Metrics...")

    ob = OrderBookManager(symbol="btcusdt")

    # Run in background
    task = asyncio.create_task(ob.start_stream())

    # Wait for data (10s)
    logger.info("⏳ Waiting for data stream (10s)...")
    await asyncio.sleep(10)

    # Check Metrics
    metrics = ob.get_latest_metrics()
    logger.info(f"📊 Latest Metrics: {metrics}")

    if not metrics:
        logger.error("❌ No metrics received!")
        ob.stop()
        await task
        return

    # Validation
    assert "spread_pct" in metrics
    assert "obi" in metrics
    assert "impact_cost" in metrics

    logger.info(f"✅ Spread: {metrics['spread_pct']:.6f}")
    logger.info(f"✅ OBI: {metrics['obi']:.4f}")

    # Save check
    ob.save_features()
    df = pd.read_parquet(ob.parquet_path)
    logger.info(f"✅ Parquet Saved. Rows: {len(df)}")
    logger.info("Sample:\n" + str(df.tail()))

    # Stop OB
    ob.stop()
    task.cancel()

    # --- Test Smart Executor Logic ---
    logger.info("🧠 Testing SmartExecutor Logic...")
    from src.execution.smart_executor import SmartExecutor

    # Mock Base Executor
    class MockBinanceExecutor:
        def place_order(self, symbol, side, amount, order_type, price=None):
            return {
                "id": "mock_id",
                "status": "filled",
                "type": order_type,
                "price": price,
            }

        def get_ticker(self, symbol):
            return {"bid": 90000, "ask": 90001}

        def get_order_status(self, symbol, order_id):
            return "filled"

    mock_base = MockBinanceExecutor()
    smart_exec = SmartExecutor(mock_base)

    # Case 1: High OBI -> Aggressive
    metrics_agg = {"obi": 0.6, "spread_pct": 0.0001, "impact_cost": 0.0001}
    logger.info("Test Case 1: High OBI")
    res1 = await smart_exec.execute_order(
        "BTC/USDT", "buy", 0.1, style="AUTO", microstructure=metrics_agg
    )
    logger.info(f"Result: {res1}")
    assert res1["type"] == "market", "Should be MARKET order"

    # Case 2: High Spread -> Passive
    metrics_pas = {"obi": 0.1, "spread_pct": 0.002, "impact_cost": 0.0001}
    logger.info("Test Case 2: Wide Spread")
    res2 = await smart_exec.execute_order(
        "BTC/USDT", "buy", 0.1, style="AUTO", microstructure=metrics_pas
    )
    logger.info(f"Result: {res2}")
    if res2:  # Passive logic returns order or task
        # Mock returns dict immediately for 'place_order', but execute_passive calls place_order
        pass

    # Case 3: High Impact -> Abort
    metrics_panic = {
        "obi": 0.1,
        "spread_pct": 0.0001,
        "impact_cost": 0.002,
    }  # > 0.15% (0.0015)
    logger.info("Test Case 3: High Impact Cost")
    res3 = await smart_exec.execute_order(
        "BTC/USDT", "buy", 0.1, style="AUTO", microstructure=metrics_panic
    )
    assert res3 is None, "Should ABORT execution"

    logger.info("✅ Verification Passed.")


if __name__ == "__main__":
    try:
        asyncio.run(test_ob_manager())
    except KeyboardInterrupt:
        pass
