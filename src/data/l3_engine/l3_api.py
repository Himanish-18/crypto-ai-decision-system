
import ctypes
import os
from pathlib import Path

# Load Shared Library
LIB_PATH = Path(__file__).parent / "libl3_engine.so"

class L3Engine:
    """
    Python Wrapper for v31 C++ L3 Market Data Engine.
    Handles Incremental Book Building, Gap Detection, and PCAP Reading.
    """
    def __init__(self):
        try:
            self.lib = ctypes.CDLL(str(LIB_PATH))
            
            # Defines Return Types and Argument Types
            self.lib.l3_create.restype = ctypes.c_void_p
            self.lib.l3_create.argtypes = []

            self.lib.l3_add.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_char, ctypes.c_double, ctypes.c_double]
            self.lib.l3_add.restype = None

            self.lib.l3_cancel.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            self.lib.l3_cancel.restype = None

            self.lib.l3_get_bid.argtypes = [ctypes.c_void_p]
            self.lib.l3_get_bid.restype = ctypes.c_double
            
            self.lib.l3_get_ask.argtypes = [ctypes.c_void_p]
            self.lib.l3_get_ask.restype = ctypes.c_double
            
            self.lib.seq_create.restype = ctypes.c_void_p
            self.lib.seq_check.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            self.lib.seq_check.restype = ctypes.c_bool
            
            self.lib.seq_get_drops.argtypes = [ctypes.c_void_p]
            self.lib.seq_get_drops.restype = ctypes.c_uint64

            self.lib.l3_free.argtypes = [ctypes.c_void_p]
            self.lib.l3_free.restype = None

            self.lib.seq_free.argtypes = [ctypes.c_void_p]
            self.lib.seq_free.restype = None
            
            # Initialize C++ Objects
            self.book = self.lib.l3_create()
            self.seq = self.lib.seq_create()
            
            print("init: L3 Engine Loaded Successfully.")
            
        except OSError as e:
            print(f"error: Failed to load libl3_engine.so: {e}")
            raise

    def add_order(self, order_id: int, side: str, price: float, size: float):
        c_side = ctypes.c_char(side.encode('utf-8'))
        self.lib.l3_add(self.book, ctypes.c_uint64(order_id), c_side, ctypes.c_double(price), ctypes.c_double(size))

    def cancel_order(self, order_id: int):
        self.lib.l3_cancel(self.book, ctypes.c_uint64(order_id))

    def get_best_bid(self) -> float:
        return self.lib.l3_get_bid(self.book)

    def get_best_ask(self) -> float:
        return self.lib.l3_get_ask(self.book)

    def processed_packet(self, seq_num: int) -> bool:
        """Returns False if a gap was detected."""
        return self.lib.seq_check(self.seq, ctypes.c_uint64(seq_num))

    def get_drops(self) -> int:
        return self.lib.seq_get_drops(self.seq)

    def __del__(self):
        if hasattr(self, 'lib'):
            self.lib.l3_free(self.book)
            self.lib.seq_free(self.seq)

if __name__ == "__main__":
    # Test
    engine = L3Engine()
    engine.add_order(1, 'B', 50000.0, 1.0)
    bid = engine.get_best_bid()
    print(f"Test Bid: {bid}")
    assert bid == 50000.0
