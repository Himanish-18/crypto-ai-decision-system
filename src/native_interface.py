import ctypes
import os
import sys
from typing import Tuple

# Load Rust Library
# Heuristic to find the lib
# On Mac: .dylib, Linux: .so
if sys.platform == "darwin":
    ext = "dylib"
else:
    ext = "so"

rust_path = os.path.join(os.path.dirname(__file__), "rust_engine", "target", "release", f"librust_engine.{ext}")
cpp_path = os.path.join(os.path.dirname(__file__), "cpp_executor", "execution_native.so")

# Fallback for missing libs
class MockLib:
    def calculate_microprice(self, b, a, bq, aq): return (b+a)/2
    def orderbook_new(self): return None
    def check_risk(self, p, q, s): return True

try:
    rust_lib = ctypes.CDLL(rust_path)
    rust_lib.calculate_microprice.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    rust_lib.calculate_microprice.restype = ctypes.c_double
    
    rust_lib.calculate_imbalance.argtypes = [ctypes.c_double, ctypes.c_double]
    rust_lib.calculate_imbalance.restype = ctypes.c_double
    
    rust_lib.orderbook_new.restype = ctypes.c_void_p
    rust_lib.orderbook_get_best_bid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

except Exception as e:
    print(f"⚠️ WARNING: Could not load Rust Engine: {e}")
    rust_lib = MockLib()

try:
    cpp_lib = ctypes.CDLL(cpp_path)
    # void init_risk_engine(double max_exp, double max_dd)
    cpp_lib.init_risk_engine.argtypes = [ctypes.c_double, ctypes.c_double]
    
    # bool check_risk(double price, double qty, int side)
    cpp_lib.check_risk.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
    cpp_lib.check_risk.restype = ctypes.c_bool
    
    # void update_position(double price, double qty)
    cpp_lib.update_position.argtypes = [ctypes.c_double, ctypes.c_double]

except Exception as e:
    print(f"⚠️ WARNING: Could not load C++ Executor: {e}")
    cpp_lib = MockLib()


class NativeEngine:
    """
    Unified Interface for HFT Native Layer.
    """
    def __init__(self):
        self.ob_ptr = rust_lib.orderbook_new() if rust_lib != MockLib() else None
        # Init Risk Engine with 100k cap, 2k drawdown
        if cpp_lib != MockLib():
            cpp_lib.init_risk_engine(100000.0, 2000.0)

    def get_microprice(self, b, a, bq, aq):
        return rust_lib.calculate_microprice(b, a, bq, aq)
    
    def check_risk_fast(self, price, qty, side_int):
        # side: 1 buy, -1 sell
        if cpp_lib == MockLib(): return True
        return cpp_lib.check_risk(price, qty, side_int)

    def update_book(self, side, price_str, qty):
        # Rust OrderBook update binding would go here
        # Need simpler C-interface for string passing if used heavily.
        pass
    
    def get_best_bid(self):
        if not self.ob_ptr: return (0.0, 0.0)
        p = ctypes.c_double()
        q = ctypes.c_double()
        rust_lib.orderbook_get_best_bid(self.ob_ptr, ctypes.byref(p), ctypes.byref(q))
        return (p.value, q.value)
