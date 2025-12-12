
import pandas as pd
import numpy as np
from itertools import combinations

# v39 Dynamic Feature Generator
# Automatically constructs polynomial, interaction, and entropy features.

class AutoFeatureGen:
    def __init__(self):
        pass
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 1. Interaction Features (A * B)
        for c1, c2 in combinations(numeric_cols, 2):
            if "price" in c1 and "vol" in c2:
                df_new[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                
        # 2. Rate of Change (Velocity)
        for col in numeric_cols:
             df_new[f"{col}_roc"] = df[col].pct_change()
             
        # 3. Log Entropy (Rolling)
        if "close" in df.columns:
             roll = df["close"].rolling(window=20)
             # Approx entropy via std dev logic
             df_new["entropy_proxy"] = np.log(roll.std() + 1e-9)
             
        return df_new.fillna(0)
