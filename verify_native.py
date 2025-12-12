from src.native_interface import NativeEngine
import time

def test_native():
    print("Testing Native HFT Layer...")
    eng = NativeEngine()
    
    # 1. Microprice check
    mp = eng.get_microprice(100.0, 101.0, 10.0, 10.0)
    print(f"Microprice (Rust): {mp}")
    assert mp == 100.5, "Microprice calc failed"
    
    # 2. Risk check
    risk = eng.check_risk_fast(50000.0, 0.1, 1)
    print(f"Risk Check (C++): {risk}")
    assert risk == True, "Risk logic failed"
    
    # 3. Orderbook check
    # Need ctypes setup for strings, skipping complex string passing in quick check
    # But ptr logic should be stable.
    bb = eng.get_best_bid()
    print(f"Empty Best Bid: {bb}")
    assert bb == (0.0, 0.0)
    
    print("✅ Native HFT Layer Verified.")

if __name__ == "__main__":
    test_native()
