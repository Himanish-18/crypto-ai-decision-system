from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api import deps
from app.db import models
from app.schemas import trade as trade_schema

router = APIRouter()


@router.get("/", response_model=List[trade_schema.Trade])
def read_trades(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Retrieve trades for current user.
    """
    trades = (
        db.query(models.Trade)
        .filter(models.Trade.user_id == current_user.id)
        .order_by(models.Trade.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return trades


@router.post("/", response_model=trade_schema.Trade)
def create_trade(
    *,
    db: Session = Depends(deps.get_db),
    trade_in: trade_schema.TradeCreate,
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Create new trade (Simulate execution).
    """
    trade = models.Trade(**trade_in.dict(), user_id=current_user.id)
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade
