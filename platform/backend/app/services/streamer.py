import asyncio
import json
import logging

import websockets

from app.api.websockets import manager

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_1m"


class MarketStreamer:
    def __init__(self):
        self.running = False

    async def start(self):
        self.running = True
        logger.info("Starting Market Streamer...")
        while self.running:
            try:
                async with websockets.connect(BINANCE_WS_URL) as ws:
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        # Process and format data
                        # (Simplified for demo, real app would parse kline)
                        payload = {"type": "market_data", "data": data}

                        await manager.broadcast(json.dumps(payload))
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(5)  # Reconnect delay

    def stop(self):
        self.running = False


streamer = MarketStreamer()
