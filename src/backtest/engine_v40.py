
import logging
import time
import pandas as pd
from src.infrastructure.messaging.event_bus import event_bus
from src.services.base_service import MicroService

# v40 Backtesting: Event-Driven Engine
# Replays market data as events over the bus to simulate realistic async behavior.

class EventDrivenBacktester(MicroService):
    def __init__(self):
        super().__init__("backtester_v40")
        self.data_feed = []
        self.current_time = None
        
    def load_data(self, data: pd.DataFrame):
        self.data_feed = data.to_dict('records')
        self.logger.info(f"Loaded {len(self.data_feed)} events for backtest.")

    def run(self):
        self.logger.info("🎬 Starting Event-Driven Backtest...")
        
        for event in self.data_feed:
            self.current_time = event.get('timestamp')
            
            # Publish Market Data Event
            if 'price' in event:
                self.publish("market_data.tick", event)
            elif 'order_book' in event:
                self.publish("market_data.book_update", event)
                
            # Simulate Processing Latency
            # time.sleep(0.0001) 
            
        self.logger.info("🏁 Backtest Complete.")
        self.publish("backtest.complete", {"status": "success"})

    def on_start(self):
        pass

    def on_stop(self):
        pass
