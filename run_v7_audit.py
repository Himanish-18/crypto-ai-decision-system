import pandas as pd
import numpy as np
from src.features.orderflow import OrderFlowFeatures
from src.models.spread_cnn import SpreadCNN
from src.models.ppo_policy import PPOPolicy

def run_v7_audit():
    print("🚀 Starting Pro v7 Intelligence Audit...")
    
    # 1. Order Flow Features
    print("\n🔍 Testing Order Flow Features...")
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
        "open": [99, 100, 101, 102, 103],
        "high": [105, 105, 105, 105, 105],
        "low": [90, 90, 90, 90, 90],
        "volume": [100, 200, 300, 400, 500],
        # Simulate taker buy vol for CVD
        "taker_buy_base_asset_volume": [60, 150, 200, 350, 480] # High buying
    })
    
    of = OrderFlowFeatures()
    df_feat = of.calculate_features(df)
    
    print("Combined Cols:", [c for c in df_feat.columns if "feat_" in c])
    latest = df_feat.iloc[-1]
    print(f"CVD: {latest['feat_cvd']:.2f} | Imbalance: {latest['feat_imbalance']:.2f}")
    
    # 2. Spread CNN
    print("\n🔍 Testing Spread CNN...")
    cnn = SpreadCNN()
    # High Vol scenario
    df_high_vol = df.copy()
    tight_prob = cnn.predict(df_high_vol)
    mode = cnn.get_execution_mode(df_high_vol)
    print(f"Prediction (Tight Prob): {tight_prob:.4f} | Exec Mode: {mode}")
    
    # 3. PPO Policy
    print("\n🔍 Testing PPO Policy...")
    ppo = PPOPolicy()
    state = {
        "prob": 0.8, # High confident
        "volatility": 0.005, # Low vol
        "trend_depth": 0.9, # Strong trend
        "panic_score": 0.0 # No panic
    }
    action = ppo.get_action(state)
    print(f"State: {state}")
    print(f"PPO Action (Size Scalar): {action:.4f}x")
    
    # Check Panic Veto in PPO
    state_panic = state.copy()
    state_panic["panic_score"] = 0.9
    action_panic = ppo.get_action(state_panic)
    print(f"Panic State Action: {action_panic:.4f}x")
    
    if action > 1.0 and action_panic < 0.2:
        print("✅ PPO logic confirmed (Aggressive in Trend, Defensive in Panic).")
    else:
        print("⚠️ PPO logic verification failed.")

if __name__ == "__main__":
    run_v7_audit()
