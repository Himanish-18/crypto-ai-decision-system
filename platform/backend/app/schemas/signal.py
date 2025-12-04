from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SignalBase(BaseModel):
    symbol: str
    direction: str
    confidence: float
    model_version: str
    features_hash: Optional[str] = None

class SignalCreate(SignalBase):
    pass

class Signal(SignalBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True
