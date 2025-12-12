
import collections
import time
from dataclasses import dataclass

# v41 Sim: L3 OrderBook
# Manages full depth with individual order IDs. Supports matching Logic.

@dataclass
class SimOrder:
    order_id: str
    side: str # 'B' or 'S'
    price: float
    size: float
    entry_time: float
    owner_id: str

class SimOrderBookL3:
    def __init__(self):
        self.bids = collections.defaultdict(list) # Price -> List[SimOrder]
        self.asks = collections.defaultdict(list)
        self.order_map = {} # order_id -> SimOrder
        self.best_bid = 0.0
        self.best_ask = float('inf')

    def add_order(self, order: SimOrder) -> list:
        """
        Adds order to book. Returns list of fills (SimFill) if crossing happens.
        For Sim simplicity, we assume 'limit' orders mainly. 
        For 'market' orders, we just match immediately against opposite book.
        """
        fills = []
        remaining_size = order.size
        
        # Matching Logic
        if order.side == 'B':
            while remaining_size > 0 and self.best_ask <= order.price:
                best_price_orders = self.asks[self.best_ask]
                while best_price_orders and remaining_size > 0:
                    match_order = best_price_orders[0]
                    trade_size = min(remaining_size, match_order.size)
                    
                    fills.append({
                        "maker_id": match_order.owner_id,
                        "taker_id": order.owner_id,
                        "price": match_order.price,
                        "size": trade_size,
                        "maker_oid": match_order.order_id
                    })
                    
                    match_order.size -= trade_size
                    remaining_size -= trade_size
                    
                    if match_order.size <= 1e-9:
                        best_price_orders.pop(0)
                        del self.order_map[match_order.order_id]
                
                if not best_price_orders:
                    del self.asks[self.best_ask]
                    self.best_ask = min(self.asks.keys()) if self.asks else float('inf')
        
        else: # Sell Side
            while remaining_size > 0 and self.best_bid >= order.price:
                best_price_orders = self.bids[self.best_bid]
                while best_price_orders and remaining_size > 0:
                    match_order = best_price_orders[0]
                    trade_size = min(remaining_size, match_order.size)
                    
                    fills.append({
                        "maker_id": match_order.owner_id,
                        "taker_id": order.owner_id,
                        "price": match_order.price,
                        "size": trade_size,
                        "maker_oid": match_order.order_id
                    })
                    
                    match_order.size -= trade_size
                    remaining_size -= trade_size
                    
                    if match_order.size <= 1e-9:
                        best_price_orders.pop(0)
                        del self.order_map[match_order.order_id]

                if not best_price_orders:
                    del self.bids[self.best_bid]
                    self.best_bid = max(self.bids.keys()) if self.bids else 0.0

        # Post remainder to book
        if remaining_size > 1e-9:
            order.size = remaining_size
            self.order_map[order.order_id] = order
            if order.side == 'B':
                self.bids[order.price].append(order)
                self.best_bid = max(self.best_bid, order.price)
            else:
                self.asks[order.price].append(order)
                self.best_ask = min(self.best_ask, order.price)
                
        return fills

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.order_map:
            return False
            
        order = self.order_map[order_id]
        if order.side == 'B':
            orders = self.bids[order.price]
            if order in orders:
                orders.remove(order)
                if not orders:
                    del self.bids[order.price]
                    self.best_bid = max(self.bids.keys()) if self.bids else 0.0
        else:
            orders = self.asks[order.price]
            if order in orders:
                orders.remove(order)
                if not orders:
                    del self.asks[order.price]
                    self.best_ask = min(self.asks.keys()) if self.asks else float('inf')
                    
        del self.order_map[order_id]
        return True

    def get_snapshot(self):
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.best_ask - self.best_bid if self.best_ask < float('inf') else 0.0
        }
