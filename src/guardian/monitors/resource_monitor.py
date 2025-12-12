import os
import time
from datetime import datetime
from src.guardian.monitors.base_monitor import BaseMonitor, Anomaly

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class ResourceMonitor(BaseMonitor):
    """
    Monitors System Resources (CPU Load, RAM Usage).
    """
    def __init__(self, cpu_limit=90.0, ram_limit_mb=2000):
        super().__init__(name="ResourceMonitor")
        self.cpu_limit = cpu_limit
        self.ram_limit_mb = ram_limit_mb

    def check(self):
        anomalies = []
        
        if not HAS_PSUTIL:
            # Fallback for CPU on Unix
            try:
                load1, _, _ = os.getloadavg()
                # Load > Core Count? Assuming 4 cores as warning threshold stub
                if load1 > 4.0:
                     anomalies.append(Anomaly(
                        type="HIGH_LOAD",
                        severity="WARNING",
                        details={"load_1min": load1},
                        timestamp=datetime.now()
                    ))
            except:
                pass
            return anomalies

        # PSUTIL Implementation
        cpu_pct = psutil.cpu_percent(interval=None) # Non-blocking
        mem = psutil.virtual_memory()
        
        if cpu_pct > self.cpu_limit:
            anomalies.append(Anomaly(
                type="HIGH_CPU",
                severity="WARNING",
                details={"cpu_percent": cpu_pct},
                timestamp=datetime.now()
            ))

        mem_used_mb = mem.used / (1024 * 1024)
        if mem_used_mb > self.ram_limit_mb:
            anomalies.append(Anomaly(
                type="MEMORY_LEAK",
                severity="CRITICAL",
                details={"used_mb": mem_used_mb, "limit": self.ram_limit_mb},
                timestamp=datetime.now()
            ))

        return anomalies
