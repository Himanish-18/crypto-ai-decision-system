from prometheus_client import start_http_server, Summary, Gauge, Counter
import time

class MonitoringService:
    """
    Exposes real-time metrics for Prometheus scraping.
    """
    def __init__(self, port: int = 8000):
        # Metrics
        self.latency_summary = Summary('trade_latency_seconds', 'Time spent processing trade')
        self.slippage_gauge = Gauge('trade_slippage_pct', 'Slippage observable in fill')
        self.pnl_gauge = Gauge('portfolio_pnl', 'Total Portfolio PnL')
        self.drift_counter = Counter('model_drift_events', 'Number of times drift detected')
        
        # Start Server
        try:
            start_http_server(port)
            print(f"📡 Prometheus Metrics Server running on port {port}")
        except Exception as e:
            print(f"⚠️ Failed to start Prometheus server: {e}")
            
    def record_latency(self, seconds: float):
        self.latency_summary.observe(seconds)
        
    def record_slippage(self, slippage: float):
        self.slippage_gauge.set(slippage)
        
    def record_pnl(self, pnl: float):
        self.pnl_gauge.set(pnl)
