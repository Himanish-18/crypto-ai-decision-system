from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import time

@dataclass(kw_only=True)
class Event:
    """Base Event Class"""
    event_id: str
    timestamp: float = field(default_factory=time.time)
    
@dataclass(kw_only=True)
class MarketTick(Event):
    """L1/L2 Market Data Update"""
    symbol: str
    price: float
    volume: float
    bid: float
    ask: float
    exchange: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class SignalEvent(Event):
    """Strategy Signal"""
    symbol: str
    signal_type: str # 'ALPHA', 'HEDGE'
    direction: str # 'BUY', 'SELL', 'HOLD'
    strength: float # 0.0 to 1.0
    confidence: float
    source: str
    features: Dict[str, float] = field(default_factory=dict)

@dataclass(kw_only=True)
class RiskCheckEvent(Event):
    """Risk Engine Response"""
    signal_id: str
    approved: bool
    adjusted_size: float
    reason: str

@dataclass(kw_only=True)
class OrderEvent(Event):
    """Execution Order Request"""
    symbol: str
    order_type: str # 'MARKET', 'LIMIT', 'TWAP', 'ICEBERG'
    side: str
    quantity: float
    price: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass(kw_only=True)
class FillEvent(Event):
    """Trade Confirmation"""
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    execution_id: str
    exchange: str
