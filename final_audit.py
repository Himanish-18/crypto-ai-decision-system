import pandas as pd
import pickle
import json
import os
from pathlib import Path
import numpy as np

DATA_DIR = Path("data")

def check_file(path):
    if not path.exists():
        print(f"❌ MISSING: {path}")
        return False
    return True

def audit_csv(path):
    if not check_file(path): return
    try:
        df = pd.read_csv(path)
        print(f"✅ CSV: {path} | Shape: {df.shape}")
        print(f"   Cols: {list(df.columns[:5])}...")
        print(f"   NaNs: {df.isna().sum().sum()}")
        if "timestamp" in df.columns:
            print(f"   Time: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}")

def audit_parquet(path):
    if not check_file(path): return
    try:
        df = pd.read_parquet(path)
        print(f"✅ PARQUET: {path} | Shape: {df.shape}")
        print(f"   Cols: {list(df.columns[:5])}...")
        print(f"   NaNs: {df.isna().sum().sum()}")
        if "timestamp" in df.columns:
            print(f"   Time: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}")

def audit_pkl(path):
    if not check_file(path): return
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"✅ PKL: {path} | Type: {type(obj)}")
        if hasattr(obj, "n_features_in_"):
            print(f"   Features Expected: {obj.n_features_in_}")
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}")

def audit_jsonl(path):
    if not check_file(path): return
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        print(f"✅ JSONL: {path} | Lines: {len(lines)}")
        if lines:
            sample = json.loads(lines[-1])
            print(f"   Last Entry: {sample.keys()}")
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}")

print("--- AUDIT START ---")
audit_csv(DATA_DIR / "clean/timeseries_clean_1H.csv")
audit_parquet(DATA_DIR / "features/features_1H_advanced.parquet")
audit_pkl(DATA_DIR / "models/model_xgb_v1.pkl")
audit_pkl(DATA_DIR / "models/scaler_v1.pkl")
audit_jsonl(DATA_DIR / "execution/logs/trading_log.jsonl")
print("--- AUDIT END ---")
