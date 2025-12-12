
import logging
from src.execution.queue.aqpe import AQPE as BaseAQPE

logger = logging.getLogger("queue_estimator_v43")

class QueuePositionEstimator(BaseAQPE):
    """
    v43 Enhanced Queue Position Estimator.
    Wraps v39 AQPE and adds Multi-Level tracking and Latency adjustment.
    """
    def __init__(self, initial_queue_size=100.0, latency_ms=50.0):
        super().__init__(initial_queue_size)
        self.latency_ms = latency_ms
        self.estimated_rank_percentile = 1.0 # 1.0 = Back of queue, 0.0 = Front
        self.start_size = initial_queue_size

    def update_market_event(self, event_type: str, qty: float, side: str):
        """
        Unified event handler.
        event_type: "TRADE", "CANCEL", "ADD"
        """
        if event_type == "TRADE":
            filled, filled_amount = self.on_trade(qty, side)
            self._update_metrics()
            return filled
            
        elif event_type == "CANCEL":
            # Advanced logic: Cancels at front vs back?
            # We assume uniform distribution for now (inherited from AQPE default 0.5)
            self.on_cancel(qty)
            self._update_metrics()
        
        elif event_type == "ADD":
             # Adds usually go behind us, unless we are modifying?
             # Passive additions don't affect our position in FIFO.
             pass
             
        return False

    def _update_metrics(self):
        if self.start_size > 0:
            self.estimated_rank_percentile = self.queue_position / self.start_size
        else:
            self.estimated_rank_percentile = 0.0

    def get_fill_likelihood(self) -> float:
        """
        Returns heuristic 0.0-1.0 score of how close we are to fill.
        """
        if self.queue_position <= 0:
            return 1.0
        # If we are in the front 10%, high likelihood
        return max(0.0, 1.0 - self.estimated_rank_percentile)
