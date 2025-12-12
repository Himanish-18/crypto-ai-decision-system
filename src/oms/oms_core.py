
import logging
import time
from typing import Dict
from .order_state import OrderState, OrderType
from .algo_slicer import AlgoSlicer

# v36 Institutional OMS Core
# Manages Order Lifecycle and Strategy Routing

class OMSCore:
    def __init__(self):
        self.logger = logging.getLogger("oms")
        self.parent_orders = {}
        self.child_orders = {}
        self.slicer = AlgoSlicer()
        
    def submit_order(self, order_req):
        order_type = order_req["type"]
        
        if order_type == OrderType.Parent_TWAP:
            self.logger.info(f"Received Parent TWAP: {order_req['qty']} @ {order_req['duration']}s")
            self.parent_orders[order_req["id"]] = order_req
            
            # Slice
            children = self.slicer.slice_twap(order_req)
            self.logger.info(f"Generated {len(children)} child slices.")
            
            # (In real sys: schedule execution)
            return len(children)
            
        else:
            # Atomic Order
            self.child_orders[order_req["id"]] = order_req
            self.logger.info(f"Received Atomic Order: {order_req['qty']}")
            return 1

    def on_execution_report(self, report):
        # Update State
        oid = report.order_id
        if oid in self.child_orders:
            self.logger.info(f"Fill for {oid}: {report.filled_qty} @ {report.price}")
            
        # If Parent involved, update parent progress...
