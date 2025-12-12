
import unittest
from src.sim.orderbook_l3 import SimOrderBookL3, SimOrder

class TestSimOrderMatching(unittest.TestCase):
    def setUp(self):
        self.book = SimOrderBookL3()

    def test_limit_order_add(self):
        order = SimOrder("1", "B", 100.0, 1.0, 0, "A")
        self.book.add_order(order)
        self.assertIn(100.0, self.book.bids)
        self.assertEqual(self.book.best_bid, 100.0)

    def test_matching_partial_fill(self):
        # Sell 1.0 @ 101
        self.book.add_order(SimOrder("1", "S", 101.0, 1.0, 0, "A"))
        
        # Buy 0.4 @ 101 -> Match
        order_buy = SimOrder("2", "B", 101.0, 0.4, 0, "B")
        fills = self.book.add_order(order_buy)
        
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]['size'], 0.4)
        self.assertEqual(fills[0]['price'], 101.0)
        
        # Clean up check
        self.assertEqual(self.book.asks[101.0][0].size, 0.6)

    def test_price_time_priority(self):
        # 1. Sell 1.0 @ 101 (Time 0)
        self.book.add_order(SimOrder("1", "S", 101.0, 1.0, 0, "A"))
        # 2. Sell 1.0 @ 101 (Time 1)
        self.book.add_order(SimOrder("2", "S", 101.0, 1.0, 1, "B"))
        
        # Buy 1.5 @ 102
        order_buy = SimOrder("3", "B", 102.0, 1.5, 2, "C")
        fills = self.book.add_order(order_buy)
        
        self.assertEqual(len(fills), 2)
        # Should match Time 0 order first
        self.assertEqual(fills[0]['maker_oid'], "1")
        self.assertEqual(fills[1]['maker_oid'], "2")
