from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger("sentinel.monitor")

@dataclass
class Anomaly:
    """
    Standard event format for detected system issues.
    """
    type: str  # e.g., "LOG_ERROR", "NEUTRAL_LOCK", "HIGH_CPU"
    severity: str  # "WARNING", "CRITICAL"
    details: Dict[str, Any]
    timestamp: datetime = datetime.now()

class BaseMonitor(ABC):
    """
    Abstract Base Class for all Sentinel Monitors.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"sentinel.monitor.{name}")

    @abstractmethod
    def check(self) -> List[Anomaly]:
        """
        Run the check and return a list of detected anomalies.
        """
        pass
