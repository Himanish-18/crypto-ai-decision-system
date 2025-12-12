
from enum import Enum, auto

# v36 OMS State Machine
# Strict definition of order lifecycles

class OrderState(Enum):
    PENDING_NEW = auto()
    NEW = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    PENDING_CANCEL = auto()
    CANCELED = auto()
    REJECTED = auto()

class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()
    Parent_TWAP = auto()
    Parent_VWAP = auto()

class ExecutionReport:
    def __init__(self, order_id, status, filled_qty, price):
        self.order_id = order_id
        self.status = status
        self.filled_qty = filled_qty
        self.price = price
