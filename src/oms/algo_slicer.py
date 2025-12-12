
import time
import uuid
from .order_state import OrderType

# v36 Algo Slicer
# Breaks Parent orders into Child orders

class AlgoSlicer:
    def __init__(self):
        pass
        
    def slice_twap(self, parent_order):
        """
        Values: TotalQty, DurationSec, IntervalSec
        Output: List of Child Orders with start times
        """
        total_qty = parent_order["qty"]
        duration = parent_order["duration"]
        interval = parent_order.get("interval", 10) # 10s default
        
        num_slices = duration // interval
        slice_qty = total_qty / num_slices
        
        slices = []
        now = time.time()
        
        for i in range(num_slices):
            child = {
                "parent_id": parent_order["id"],
                "child_id": str(uuid.uuid4()),
                "qty": slice_qty,
                "type": OrderType.LIMIT,
                "trigger_time": now + (i * interval)
            }
            slices.append(child)
            
        return slices
