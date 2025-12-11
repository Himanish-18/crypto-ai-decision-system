import asyncio
import json
import logging
import time
import pandas as pd
import numpy as np
import threading
from datetime import datetime, timezone
import websockets
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger("orderbook_features")

class OrderBookManager:
    """
    Manages real-time Order Book connection and feature computation.
    """
    def __init__(self, symbol: str = "btcusdt", levels: int = 10):
        self.symbol = symbol.lower()
        self.levels = levels
        self.ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@depth20@100ms"
        self.latest_book: Dict = {}
        self.running = False
        self.metrics: Dict = {}
        
        # Persistence
        self.data_dir = Path(__file__).resolve().parents[2] / "data" / "features"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.data_dir / "orderbook_features.parquet"
        self.parquet_path = self.data_dir / "orderbook_features.parquet"
        self.history: List[Dict] = []
        
        # Market Pulse State
        self.last_pulse_ts = 0
        self.last_pulse_price = None
        
        # HFT Listeners (Data Wiring)
        self.listeners = []
        
        # Persistence Logic
        self.total_updates = 0
        self.lock = threading.Lock() # Fix Race Condition
        
    def register_listener(self, callback):
        self.listeners.append(callback)
        
    async def start_stream(self):
        """Start the WebSocket stream."""
        self.running = True
        logger.info(f"🔌 Connecting to OB Stream: {self.ws_url}")
        
        while self.running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("✅ Connected to Binance Depth Stream")
                    
                    while self.running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        self._process_depth_update(data)
                        
                        # Periodically save to parquet (updates-based, not length-based)
                        if self.total_updates % 100 == 0 and self.total_updates > 0:
                            self.save_features()
                            
            except Exception as e:
                logger.error(f"⚠️ WS Connection Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    def _process_depth_update(self, data: Dict):
        """
        Process depth update and compute metrics.
        Binance Futures Depth Payload:
        {
          "e": "depthUpdate",
          "E": 123456789,
          "T": 123456788,
          "s": "BTCUSDT",
          "b": [["Price", "Qty"], ...],
          "a": [["Price", "Qty"], ...]
        }
        """
        try:
            timestamp = pd.to_datetime(data["T"], unit="ms", utc=True)
            bids = np.array(data["b"], dtype=float)
            asks = np.array(data["a"], dtype=float)
            
            if len(bids) == 0 or len(asks) == 0:
                return

            # Notify Listeners (HFT Wiring)
            for listener in self.listeners:
                try:
                    listener(bids, asks)
                except Exception as e:
                    logger.error(f"Listener error: {e}")

            # Sort just in case (Bids desc, Asks asc)
            # Binance stream usually ordered, but safe to valid
            # bids = bids[bids[:,0].argsort()[::-1]]
            # asks = asks[asks[:,0].argsort()]
            
            # --- Metrics Computation ---
            
            # 1. Spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price
            
            # 2. Imbalance (OBI) - Top 5 levels
            bid_vol_5 = np.sum(bids[:5, 1])
            ask_vol_5 = np.sum(asks[:5, 1])
            obi = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5)
            
            # 3. Microstructure Noise: Weighted Imbalance (Top 10)
            # Weight by 1/rank or 1/distance
            w_bids = 0
            w_asks = 0
            for i in range(min(len(bids), 10)):
                w = 1 / (i + 1)
                w_bids += bids[i][1] * w
                w_asks += asks[i][1] * w
            weighted_imbalance = (w_bids - w_asks) / (w_bids + w_asks)
            
            # 4. Impact Cost (Slippage for $10k order)
            TARGET_SIZE_USD = 10000
            
            def calc_impact(side_depth, target_usd):
                filled_usd = 0
                total_qty = 0
                avg_price = 0
                
                for p, q in side_depth:
                    amount_usd = p * q
                    if filled_usd + amount_usd >= target_usd:
                        needed_usd = target_usd - filled_usd
                        needed_qty = needed_usd / p
                        filled_usd += needed_usd
                        total_qty += needed_qty
                        # Weighted avg price
                        # Crude approx: we perform X qty at P
                        # Actually we want (VWAP - Mid) / Mid
                        break
                    else:
                        filled_usd += amount_usd
                        total_qty += q
                
                if total_qty == 0: return 0.0
                
                # Simplified Impact: (WorstFillPrice - BestPrice) / BestPrice
                # Or (VWAP_of_fill - Mid) / Mid
                # Let's take price of last fill
                last_fill_price = p
                return abs(last_fill_price - mid_price) / mid_price

            impact_bid = calc_impact(bids, TARGET_SIZE_USD)
            impact_ask = calc_impact(asks, TARGET_SIZE_USD)
            impact_cost = (impact_bid + impact_ask) / 2
            
            # 5. Liquidity Ratio (Bids vs Asks total visible)
            total_bid_liq = np.sum(bids[:, 1])
            total_ask_liq = np.sum(asks[:, 1])
            total_bid_liq = np.sum(bids[:, 1])
            total_ask_liq = np.sum(asks[:, 1])
            liquidity_ratio = total_bid_liq / (total_ask_liq + 1e-9)
            
            # --- Market Pulse Logger (Every 1s) ---
            now_ts = time.time()
            if now_ts - self.last_pulse_ts >= 1.0:
                if self.last_pulse_price is not None:
                    delta = mid_price - self.last_pulse_price
                    direction = "⬆️" if delta > 0 else "⬇️" if delta < 0 else "➡️"
                    color_icon = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
                    
                    # Log only if there is legitimate connection/activity
                    logger.info(f"{color_icon} Pulse: {mid_price:.2f} | {direction} {delta:+.2f} USD")
                
                self.last_pulse_price = mid_price
                self.last_pulse_ts = now_ts
            
            metrics = {
                "timestamp": timestamp,
                "spread_pct": spread_pct,
                "obi": obi,
                "weighted_imbalance": weighted_imbalance,
                "impact_cost": impact_cost,
                "liquidity_ratio": liquidity_ratio,
                "best_bid": best_bid,
                "best_ask": best_ask
            }
            
            self.metrics = metrics
            
            with self.lock:
                self.history.append(metrics)
                # Trim history in memory
                if len(self.history) > 10000:
                    self.history = self.history[-10000:]
                self.total_updates += 1
                
        except Exception as e:
            logger.error(f"Error processing depth: {e}")

    def get_latest_metrics(self) -> Dict:
        """Return the latest computed metrics."""
        return self.metrics

    def save_features(self):
        """Dump history to parquet."""
        try:
            # Snapshot with Lock
            with self.lock:
                if not self.history:
                    return
                data_snapshot = list(self.history)
                snap_len = len(data_snapshot)
            
            logger.info(f"💾 Saving Snapshot: {snap_len} records")
            
            # Save outside lock to avoid blocking HFT loop
            df = pd.DataFrame(data_snapshot)
            df.to_parquet(self.parquet_path)
        except Exception as e:
            logger.error(f"Failed to save features: {e}")

    def stop(self):
        self.running = False

# Simpler Synchronous Fetch for Pipeline (Non-Streaming)
def fetch_snapshot(symbol="BTCUSDT"):
    import requests
    url = f"https://fapi.binance.com/fapi/v1/depth?symbol={symbol}&limit=20"
    try:
        data = requests.get(url).json()
        # Mock wrapper to reuse processing logic?
        # Manually compute
        bids = np.array(data["bids"], dtype=float)
        asks = np.array(data["asks"], dtype=float)
        
        bid_vol_5 = np.sum(bids[:5, 1])
        ask_vol_5 = np.sum(asks[:5, 1])
        obi = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5)
        
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask)/2
        spread = (best_ask - best_bid) / mid
        
        return {
            "obi": obi,
            "spread_pct": spread,
            "best_bid": best_bid,
            "best_ask": best_ask
        }
    except Exception as e:
        logger.error(f"Snapshot fetch failed: {e}")
        return {}
