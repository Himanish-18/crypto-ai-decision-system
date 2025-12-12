from abc import ABC, abstractmethod
from typing import Any, Dict


class PolymorphicAgent(ABC):
    """
    Base class for v16 Polymorphic Trading Agents.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and return signal package.
        Returns:
            {
                "signal": float (-1.0 to 1.0),
                "confidence": float (0.0 to 1.0),
                "risk_budget": float (0.0 to 1.0),
                "latency_cost": float (ms, estimated),
                "meta": {}
            }
        """
        pass

    def get_name(self) -> str:
        return self.name
