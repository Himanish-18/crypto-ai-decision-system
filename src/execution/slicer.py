
import random
import logging
from typing import List, Dict

logger = logging.getLogger("slicer_v43")

class OrderSlicer:
    """
    v43 Execution Algo: Slices large parent orders into child clips.
    Supports: Iceberg (Uniform), Randomized (Variance).
    """
    def __init__(self, min_clip_size=0.01, variance=0.2):
        self.min_clip_size = min_clip_size
        self.variance = variance # +/- 20% random
        
    def slice_order(self, total_qty: float, strategy="random") -> List[float]:
        """
        Generate a list of child order sizes that sum to total_qty.
        """
        slices = []
        remaining = total_qty
        
        while remaining > self.min_clip_size:
            # Base clip
            # Ideally this comes from a "Participation Rate" or "Time Horizon"
            # For v43 Basic: Assume we want to slice into chunks approx 10% of total or fixed size?
            # Let's target 5-10 clips per order for now.
            target_clip = max(self.min_clip_size, total_qty / 8.0) 
            
            if strategy == "random":
                # Add noise
                noise = random.uniform(-self.variance, self.variance)
                clip = target_clip * (1.0 + noise)
            else:
                clip = target_clip
            
            clip = round(clip, 4)
            clip = min(clip, remaining)
            
            if clip < self.min_clip_size:
                # Too small, just take remainder
                break
                
            slices.append(clip)
            remaining -= clip
            
        if remaining > 0:
            slices.append(round(remaining, 4))
            
        logger.debug(f"Sliced {total_qty} into {len(slices)} clips: {slices}")
        return slices

    def estimate_time_horizon(self, slices: List[float], interval_sec=10) -> float:
        return len(slices) * interval_sec
