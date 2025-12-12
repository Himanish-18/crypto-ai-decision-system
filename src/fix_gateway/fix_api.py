import ctypes
import os
import sys

# Load Native FIX Library
lib_path = os.path.join(os.path.dirname(__file__), "fix_engine.so")

class MockFix:
    def fix_create(self, s, t): return 1
    def fix_logon(self, s, buf, l): pass
    def fix_new_order(self, s, sym, side, qty, px, buf, l): pass

try:
    _fix_lib = ctypes.CDLL(lib_path)
    _fix_lib.fix_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    _fix_lib.fix_create.restype = ctypes.c_void_p
    
    _fix_lib.fix_logon.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    
    _fix_lib.fix_new_order.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char, ctypes.c_double, ctypes.c_double, ctypes.c_char_p, ctypes.c_int]

except Exception as e:
    print(f"⚠️ FIX Gateway Native Load Failed: {e}")
    _fix_lib = MockFix()


class FIXClient:
    """
    Python Wrapper for Native C++ FIX Engine.
    """
    def __init__(self, sender_comp_id: str, target_comp_id: str):
        self.session = _fix_lib.fix_create(
            sender_comp_id.encode('ascii'), 
            target_comp_id.encode('ascii')
        )
        
    def logon(self) -> str:
        buf = ctypes.create_string_buffer(1024)
        _fix_lib.fix_logon(self.session, buf, 1024)
        return buf.value.decode('ascii')
    
    def new_order_single(self, symbol: str, side: str, qty: float, price: float) -> str:
        # Side: '1' buy, '2' sell
        side_char = b'1' if side.upper() == "BUY" else b'2'
        buf = ctypes.create_string_buffer(1024)
        _fix_lib.fix_new_order(
            self.session, 
            symbol.encode('ascii'), 
            ctypes.c_char(side_char), 
            ctypes.c_double(qty), 
            ctypes.c_double(price),
            buf, 
            1024
        )
        return buf.value.decode('ascii')

    def destroy(self):
         # Add destructor bindings if needed
         pass
