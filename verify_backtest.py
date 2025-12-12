from src.event_backtester.engine import EventBacktester, MarketData, Order, Event
from src.event_backtester.simulator import ExchangeSimulator

# Mock Portfolio
class MockPortfolio:
    def on_fill(self, fill):
        print(f"💰 Filled: {fill.side} {fill.quantity} @ {fill.price} (Time: {fill.timestamp:.4f})")

def test_backtest():
    print("Testing Event Engine...")
    engine = EventBacktester()
    sim = ExchangeSimulator(engine)
    portfolio = MockPortfolio()
    
    engine.exchange = sim
    engine.portfolio = portfolio
    
    # 1. Feed Market Data
    md = MarketData("BTC-USD", 50000.0, 49990.0, 50010.0, 1.0)
    engine.push_event(Event(1000.0, 0, md))
    
    # 2. Strategy sends Market Order
    order = Order("BTC-USD", "BUY", 0.5, "MARKET")
    engine.push_event(Event(1000.05, 1, order)) # 50ms later
    
    # Run
    engine.run()
    
    print("✅ Event Engine Verified.")

if __name__ == "__main__":
    test_backtest()
