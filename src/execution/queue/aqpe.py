
import logging

# v39 Adaptive Queue Position Estimator (AQPE)
# Software-based estimation of our priority in the limit order book.

class AQPE:
    def __init__(self, initial_queue_size=100.0):
        self.queue_position = initial_queue_size
        self.logger = logging.getLogger("mq_execution")
        
    def on_trade(self, trade_size, trade_side):
        """
        Updates estimated position based on trades that occur at our price level.
        """
        # If trade eats into our side of book
        if trade_size > 0:
            filled_qty = min(self.queue_position, trade_size)
            self.queue_position -= filled_qty
            
            if self.queue_position <= 0:
                 self.logger.info("AQPE: Estimated Fill! Queue Depleted.")
                 return True, filled_qty # Filled
                 
        return False, 0.0

    def on_cancel(self, cancel_qty):
        # Conservative estimate: Assume 50% of cancels were ahead of us
        self.queue_position -= (cancel_qty * 0.5)
        self.queue_position = max(0.0, self.queue_position)
        
    def reset(self, new_level_size):
        self.queue_position = new_level_size
