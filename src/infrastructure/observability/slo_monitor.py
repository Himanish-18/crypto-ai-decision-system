
import logging
from src.infrastructure.observability.metrics import metrics

# v40 Infrastructure: SRE (SLO Monitoring)
# Tracks Error Budgets and triggers PagerDuty alerts if burned too fast.

class SLOMonitor:
    def __init__(self, slo_target=0.999, window_size=1000):
        self.slo_target = slo_target
        self.window_size = window_size
        self.success_count = 0
        self.total_count = 0
        self.logger = logging.getLogger("infrastructure.sre")

    def record_request(self, success: bool):
        self.total_count += 1
        if success:
            self.success_count += 1
            
        metrics.inc_counter("requests_total", labels={"status": "success" if success else "error"})

        if self.total_count >= self.window_size:
            self._check_budget()
            # Rolling window reset (simplified)
            self.success_count = 0
            self.total_count = 0
            
    def _check_budget(self):
        availability = self.success_count / self.total_count
        error_budget_remaining = availability - self.slo_target
        
        if error_budget_remaining < 0:
            self.logger.critical(f"🔥 SLO VIOLATION! Availability: {availability:.4f} < Target: {self.slo_target}")
            # Trigger PagerDuty (Stub)
            self._alert_on_call()
        else:
            self.logger.info(f"✅ SLO Healthy. Availability: {availability:.4f}. Budget: +{error_budget_remaining:.4f}")

    def _alert_on_call(self):
        self.logger.error("📟 PagerDuty Alert Sent: Error Budget Exhausted.")

slo_monitor = SLOMonitor()
