import asyncio
import logging

from src.execution.micro_execution import MicroExecution
from src.features.microstructure_features import MicrostructureFeatures
from src.ingest.l2_stream import L2Stream
from src.models.live_inference import LiveInferenceEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main_l2")


class CryptoAlphaSystem:
    def __init__(self):
        self.stream = L2Stream(symbol="btcusdt", callbacks=[self.on_tick])
        self.feature_engine = MicrostructureFeatures()
        self.inference_engine = LiveInferenceEngine()
        self.execution_engine = MicroExecution()

    async def start(self):
        logger.info("Starting Crypto Alpha System (L2)...")
        await self.stream.connect()

    async def on_tick(self, snapshot):
        # 1. Compute Features
        features = self.feature_engine.compute_features(snapshot)
        if not features:
            return

        # 2. Inference
        signal = self.inference_engine.predict(features)

        # 3. Execution
        await self.execution_engine.execute(signal, snapshot)

        # 4. Maintenance (Cancel Stale Orders)
        await self.execution_engine.cancel_stale_orders()


if __name__ == "__main__":
    system = CryptoAlphaSystem()
    try:
        asyncio.run(system.start())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
