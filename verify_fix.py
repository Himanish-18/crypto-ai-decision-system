from src.fix_gateway.fix_api import FIXClient

def test_fix():
    print("Testing Native FIX Gateway...")
    client = FIXClient("MY_FUND", "COINBASE")
    
    # 1. Logon
    msg = client.logon()
    print(f"Logon Msg: {msg}")
    assert "35=A" in msg, "Logon failed"
    
    # 2. Order
    order = client.new_order_single("BTC-USD", "BUY", 1.5, 50000.0)
    print(f"Order Msg: {order}")
    assert "35=D" in order, "Order creation failed"
    assert "54=1" in order, "Side wrong"
    
    print("✅ FIX Native Gateway Verified.")

if __name__ == "__main__":
    test_fix()
