from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api import deps
from app.db import models
from app.schemas import signal as signal_schema

router = APIRouter()

@router.get("/", response_model=List[signal_schema.Signal])
def read_signals(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Retrieve signals.
    """
    signals = db.query(models.Signal).order_by(models.Signal.timestamp.desc()).offset(skip).limit(limit).all()
    return signals

@router.post("/", response_model=signal_schema.Signal)
def create_signal(
    *,
    db: Session = Depends(deps.get_db),
    signal_in: signal_schema.SignalCreate,
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Create new signal (Admin only or System).
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    
    signal = models.Signal(**signal_in.dict())
    db.add(signal)
    db.commit()
    db.refresh(signal)
    return signal
