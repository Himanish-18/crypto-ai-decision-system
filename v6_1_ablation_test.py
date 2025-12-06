import pandas as pd
import numpy as np
from src.intelligence.noise_filter import NoiseFilter
from src.intelligence.kalman_smoother import KalmanSmoother
from src.intelligence.vol_adaptive import VolAdaptiveThreshold
from src.intelligence.trend_depth import TrendDepth
from pathlib import Path

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "features" / "features_v5_expanded.parquet"

def run_ablation_v6_1():
    print("🚀 Starting v6.1 (TrendDepth) Ablation Test...")
    
    # Load Data
    try:
        df = pd.read_parquet(DATA_PATH)
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Test Set
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        print(f"📊 Test Data Size: {len(test_df)}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Init Components
    noise_filter = NoiseFilter()
    kalman = KalmanSmoother()
    vol_adaptive = VolAdaptiveThreshold()
    trend_depth = TrendDepth()
    
    # Simulate Data Columns for TD if missing
    if "adx" not in test_df.columns: test_df["adx"] = 50.0  # Placeholder or Calc
    
    # Generate Synthetic Probabilities (mimicking a realistic model)
    np.random.seed(42)
    # Good model in Trend, Bad in Chop
    future_ret = test_df["btc_close"].pct_change().shift(-1).fillna(0)
    
    test_df["return"] = test_df["btc_close"].pct_change()
    test_df["vol"] = test_df["return"].rolling(14).std()
    
    probs = []
    for i in range(len(test_df)):
        vol = test_df["vol"].iloc[i]
        ret = future_ret.iloc[i]
        
        # Trend Regime -> Predictive
        if vol > 0.005: 
            p = 0.5 + (np.sign(ret) * 0.15) + np.random.normal(0, 0.1)
        else: # Chop
            p = 0.5 + np.random.normal(0, 0.2) 
        probs.append(np.clip(p, 0, 1))
    test_df["prob"] = probs
    
    # --- SIMULATION ---
    equity_v6 = [1.0] # Old v6 (No TD)
    equity_v6_1 = [1.0] # New v6.1 (With TD)
    
    price = test_df["btc_close"].values
    
    for i in range(200, len(test_df)-1):
        curr_price = price[i]
        next_price = price[i+1]
        ret = (next_price - curr_price) / curr_price
        
        # Data Slice
        candles = test_df.iloc[0:i+1] # Full history up to i
        # Slicing full df is slow, pass window
        # TrendDepth needs ~200 window
        window_candles = test_df.iloc[i-201:i+1]
        
        vol = test_df["vol"].iloc[i]
        if np.isnan(vol): vol = 0.01
        prob = test_df["prob"].iloc[i]
        
        # Common v6 Steps
        smooth_prob = kalman.smooth(prob)
        is_chop = noise_filter.is_chop(test_df["btc_close"].iloc[i-30:i+1])
        adaptive_thresh_base = vol_adaptive.get_threshold(vol)
        
        # --- Logic v6 (Original) ---
        sig_v6 = 0
        if is_chop:
            if smooth_prob > 0.65: sig_v6 = 1
        else:
            if smooth_prob > adaptive_thresh_base: sig_v6 = 1
            
        # --- Logic v6.1 (Trend Depth) ---
        # Calculate TD
        # Mocking or calling real
        td_score = trend_depth.calculate(window_candles)
        
        eff_thresh = adaptive_thresh_base
        
        if td_score > 0.65:
            eff_thresh -= 0.05
        elif td_score < 0.25:
            eff_thresh += 0.04
            
        sig_v6_1 = 0
        if is_chop:
            if td_score > 0.75: # Override
                if smooth_prob > eff_thresh: sig_v6_1 = 1
            else:
                if smooth_prob > 0.65: sig_v6_1 = 1 # Standard
        else:
            if smooth_prob > eff_thresh: sig_v6_1 = 1
            
        # PnL
        if sig_v6 == 1: equity_v6.append(equity_v6[-1] * (1 + ret))
        else: equity_v6.append(equity_v6[-1])
            
        if sig_v6_1 == 1: equity_v6_1.append(equity_v6_1[-1] * (1 + ret))
        else: equity_v6_1.append(equity_v6_1[-1])
        
    # Metrics
    eq_v6 = np.array(equity_v6)
    eq_v6_1 = np.array(equity_v6_1)
    
    ret_v6 = (eq_v6[-1] - 1) * 100
    ret_v6_1 = (eq_v6_1[-1] - 1) * 100
    
    def max_dd(eq):
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        return np.min(dd) * 100
        
    dd_v6 = max_dd(eq_v6)
    dd_v6_1 = max_dd(eq_v6_1)
    
    print("\n📊 v6.1 ABLATION RESULTS")
    print("-" * 50)
    print(f"Metrics          | v6 (Prev)     | v6.1 (TrendDepth)")
    print(f"-----------------|---------------|------------------")
    print(f"Total Return     | {ret_v6:8.2f}%    | {ret_v6_1:8.2f}%")
    print(f"Max Drawdown     | {dd_v6:8.2f}%    | {dd_v6_1:8.2f}%")
    print("-" * 50)
    
    if ret_v6_1 > ret_v6:
        print("✅ SUCCESS: Trend Depth improved returns/efficiency.")
    else:
        print("⚠️ RESULT: Mixed results.")

if __name__ == "__main__":
    run_ablation_v6_1()
