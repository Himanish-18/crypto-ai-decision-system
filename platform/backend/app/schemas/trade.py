from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TradeBase(BaseModel):
    symbol: str
    side: str
    price: float
    amount: float
    pnl: Optional[float] = None


class TradeCreate(TradeBase):
    pass


class Trade(TradeBase):
    id: int
    user_id: int
    timestamp: datetime

    class Config:
        from_attributes = True
