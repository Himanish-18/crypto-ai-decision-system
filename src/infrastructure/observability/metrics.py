
import logging
import time
from typing import Dict
from collections import defaultdict

# v40 Infrastructure: Observability (Metrics)
# Prometheus-compatible metrics registry (Counters, Gauges, Histograms).

class MetricsRegistry:
    def __init__(self):
        self._counters = defaultdict(float)
        self._gauges = defaultdict(float)
        self._histograms = defaultdict(list)
        self.logger = logging.getLogger("infrastructure.metrics")

    def inc_counter(self, name: str, value: float = 1.0, labels: Dict = None):
        key = self._format_key(name, labels)
        self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Dict = None):
        key = self._format_key(name, labels)
        self._gauges[key] = value

    def observe_histogram(self, name: str, value: float, labels: Dict = None):
        key = self._format_key(name, labels)
        self._histograms[key].append(value)
        # Keep fixed size window for memory safety
        if len(self._histograms[key]) > 1000:
            self._histograms[key].pop(0)

    def _format_key(self, name: str, labels: Dict) -> str:
        if not labels: return name
        label_str = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])
        return f'{name}{{{label_str}}}'

    def scrape(self) -> str:
        """Returns Prometheus text format metrics."""
        lines = []
        for k, v in self._counters.items():
            lines.append(f"# TYPE {k.split('{')[0]} counter")
            lines.append(f"{k} {v}")
        for k, v in self._gauges.items():
            lines.append(f"# TYPE {k.split('{')[0]} gauge")
            lines.append(f"{k} {v}")
        return "\n".join(lines)

metrics = MetricsRegistry()
