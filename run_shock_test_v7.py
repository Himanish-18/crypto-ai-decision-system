import pandas as pd
import numpy as np
from src.features.sentiment_features import SentimentFeatures
from src.models.panic_exit import PanicExitModel

def run_audit():
    print("🚀 Starting v7 Shock & Sentiment Audit...")
    
    # 1. Test Feature Generation
    print("\n🔍 Testing Sentiment Features...")
    # Create sufficient history (30+) + Crash
    data = []
    for i in range(40):
         data.append({
             "close": 100, "open": 100, "high": 102, "low": 98,
             "volume": 10, "fundingRate": 0.01, "openInterest": 1000
         })
    
    # Crash at end
    data[-1] = {"close": 90, "open": 100, "high": 100, "low": 80, "volume": 500, "fundingRate": 0.05, "openInterest": 800}
    data[-2] = {"close": 99, "open": 100, "high": 100, "low": 98, "volume": 200, "fundingRate": 0.01, "openInterest": 1000}
    
    df = pd.DataFrame(data)
    
    sf = SentimentFeatures()
    df_feat = sf.calculate_proxies(df, window=3)
    
    print("Columns Generated:", [c for c in df_feat.columns if "feat_" in c])
    latest = df_feat.iloc[-1]
    print(f"Latest Fear Proxy: {latest['feat_fear_proxy']:.4f}")
    print(f"Latest Panic Proxy: {latest['feat_panic_proxy']:.4f}")
    
    # 2. Test Panic Exit Model
    print("\n🔍 Testing Panic Exit Model (PEM)...")
    pem = PanicExitModel()
    
    # Scenario: CRASH
    res = pem.predict(df_feat)
    print(f"Crash Scenario -> Score: {res['panic_score']:.2f} | Exit: {res['exit_signal']}")
    print(f"Reason: {res['reason']}")
    
    if res['exit_signal']:
        print("✅ PEM correctly detected the crash.")
    else:
        print("⚠️ PEM Failed to detect crash.")
        
    # Scenario: NORMAL
    df_normal = pd.DataFrame({
         "close": [100]*30, "open": [100]*30, "high": [101]*30, "low": [99]*30,
         "volume": [10]*30, "fundingRate": [0.001]*30, "openInterest": [1000]*30,
         "atr": [1.0]*30
    })
    df_normal = sf.calculate_proxies(df_normal)
    res_norm = pem.predict(df_normal)
    print(f"Normal Scenario -> Score: {res_norm['panic_score']:.2f} | Exit: {res_norm['exit_signal']}")
    
    if not res_norm['exit_signal']:
        print("✅ PEM correctly ignored normal market.")
    else:
        print("⚠️ PEM False Positive on normal market.")

if __name__ == "__main__":
    run_audit()
