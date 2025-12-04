from sqlalchemy.orm import Session
from app.db import base, models
from app.core.config import settings
from app.db.base import engine

def init_db(db: Session):
    # Create tables
    base.Base.metadata.create_all(bind=engine)
    
    # Create superuser if not exists
    # (In real app, use a proper migration script or CLI command)
    pass
