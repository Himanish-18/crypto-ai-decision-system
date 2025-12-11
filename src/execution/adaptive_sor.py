import logging
import random
from typing import List, Dict
from src.engine.events import OrderEvent
from src.execution.impact_model import MarketImpactModel

logger = logging.getLogger("execution.sor")

class AdaptiveSOR:
    """
    v24 Adaptive Smart Order Router.
    Slices Large Orders (Iceberg) and Routes Dynamically.
    """
    def __init__(self):
        self.impact_model = MarketImpactModel()
        
    def slice_order(self, parent_order: OrderEvent, vol_profile: float) -> List[OrderEvent]:
        """
        Break parent order into child orders based on impact model.
        """
        parent_qty = parent_order.quantity
        
        # 1. Check Impact
        # Stub volume and vol
        impact = self.impact_model.estimate_impact_bps(parent_qty, 1000000, 0.02)
        
        if impact < 2.0: # < 2bps impact is negligible
            return [parent_order]
            
        # 2. Slice needed (Simple TWAP slice)
        num_slices = int(impact) + 2 # Heuristic: more slices for higher impact
        slice_qty = parent_qty / num_slices
        
        children = []
        for i in range(num_slices):
            child = OrderEvent(
                event_id=f"{parent_order.event_id}_child_{i}",
                symbol=parent_order.symbol,
                order_type="LIMIT", # Use limit for slices
                side=parent_order.side,
                quantity=slice_qty,
                params={"parent_id": parent_order.event_id, "slice_idx": i}
            )
            children.append(child)
            
        logger.info(f"❄️ ICEBERG: Sliced {parent_qty} into {num_slices} chunks ({slice_qty:.4f} each). Expected Impact: {impact:.2f}bps")
        return children
