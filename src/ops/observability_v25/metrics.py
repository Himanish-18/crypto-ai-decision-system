import logging
from typing import Any, Dict

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger("observability_metrics")


class MetricsExporter:
    """
    Exports custom metrics to Prometheus for Grafana visualization.
    """

    def __init__(self, port: int = 8000):
        self.port = port

        # Define Metrics
        self.system_up = Gauge("system_up", "System is running")
        self.model_drift = Gauge(
            "model_drift_score", "KL Divergence score for Feature Drift"
        )
        self.prediction_latency = Histogram(
            "prediction_latency_seconds", "Time spent generating predictions"
        )
        self.trades_executed = Counter(
            "trades_executed_total", "Total trades executed", ["strategy", "side"]
        )
        self.portfolio_value = Gauge(
            "portfolio_value_usd", "Current Net Liquidation Value"
        )
        self.active_risk_cvar = Gauge("active_risk_cvar", "Current 95% cVaR exposure")

    def start_server(self):
        try:
            start_http_server(self.port)
            logger.info(f"📊 Prometheus Metrics Server started on port {self.port}")
            self.system_up.set(1)
        except Exception as e:
            logger.error(f"❌ Failed to start metrics server: {e}")

    def update_portfolio(self, value: float, cvar: float):
        self.portfolio_value.set(value)
        self.active_risk_cvar.set(cvar)

    def record_trade(self, strategy: str, side: str):
        self.trades_executed.labels(strategy=strategy, side=side).inc()

    def set_drift(self, drift_score: float):
        self.model_drift.set(drift_score)
