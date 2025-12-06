import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.models.alpha_ensemble import AlphaEnsemble
import logging

logger = logging.getLogger("multifactor_model")

class MultiFactorModel:
    """
    Multi-Factor Fusion Engine.
    Combines signals using:
    1. Rank Aggregation (Robust to outliers)
    2. Weighted Voting (Heuristic or Perf-based)
    3. Stacking (ML Ensemble)
    """
    def __init__(self):
        self.stacking_model = AlphaEnsemble()
        self.weights = {
            "rank_agg": 0.3, 
            "weighted": 0.2, 
            "stacking": 0.5
        }
        
    def train(self, df: pd.DataFrame, target_col="y_direction_up"):
        """Train regime-specific stacking models."""
        logger.info("🧠 Training Multi-Factor Regime Models...")
        
        # 1. Partition Data
        # Normal Cluster: Bull Trend, Bear Trend, Sideways
        # Crisis Cluster: High Volatility, Low Liquidity, Macro Event
        
        normal_regimes = ["Bull Trend", "Bear Trend", "Sideways", "Normal"]
        crisis_regimes = ["High Volatility", "Low Liquidity", "Macro Event"]
        
        df_normal = df[df["regime"].isin(normal_regimes)].copy()
        df_crisis = df[df["regime"].isin(crisis_regimes)].copy()
        
        logger.info(f"Training 'Normal' Model on {len(df_normal)} samples.")
        logger.info(f"Training 'Crisis' Model on {len(df_crisis)} samples.")
        
        # Initialize Models
        self.models = {}
        
        # Train Normal
        if not df_normal.empty:
            self.models["normal"] = AlphaEnsemble()
            self.models["normal"].train(df_normal, target_col)
            
        # Train Crisis (Fall back to Normal if empty? or Train on full data with weights?)
        # User requested "trained only on high-volatility segments".
        if not df_crisis.empty:
            self.models["crisis"] = AlphaEnsemble() # Could customize params here for robustness (e.g. less depth)
            self.models["crisis"].train(df_crisis, target_col)
        else:
            logger.warning("No Crisis data found! Fallback to Normal model.")
            # This assumes 'normal' model was trained. If not, self.models["normal"] would not exist.
            # A more robust approach might be to ensure 'normal' is always trained or handle its absence.
            if "normal" in self.models:
                self.models["crisis"] = self.models["normal"]
            else:
                logger.error("Neither Normal nor Crisis data available for training!")
                # Depending on desired behavior, could raise an error or initialize a dummy model
                self.models["crisis"] = AlphaEnsemble() # Initialize an empty model or handle gracefully
            
        return self

    def predict_composite_score(self, df: pd.DataFrame) -> pd.Series:
        """Generate final composite score (0-1) with Regime Routing."""
        # Ensure alpha features exist
        alpha_cols = [c for c in df.columns if "alpha_" in c]
        if not alpha_cols:
            logger.warning("No alpha columns found for Fusion Engine!")
            return pd.Series(0.5, index=df.index)

        # 1. Rank Aggregation
        ranks = df[alpha_cols].rank(pct=True, axis=0)
        score_rank_agg = ranks.mean(axis=1) # 0 to 1
        
        # 2. Weighted Voting
        score_weighted = df[alpha_cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-9), axis=0).mean(axis=1)
        score_weighted = 1 / (1 + np.exp(-score_weighted))
        
        # 3. Regime-Based Stacking Prediction
        # Route samples
        normal_regimes = ["Bull Trend", "Bear Trend", "Sideways", "Normal"]
        
        # Identify masks
        # If 'regime' col missing, assume Normal
        if "regime" not in df.columns:
            logger.warning("Regime column missing in prediction! Defaulting to Normal.")
            is_normal = pd.Series(True, index=df.index)
        else:
            is_normal = df["regime"].isin(normal_regimes)
            
        idx_normal = df.index[is_normal]
        idx_crisis = df.index[~is_normal]
        
        scores = pd.Series(index=df.index, dtype=float)
        
        # Predict Normal
        if len(idx_normal) > 0:
            if "normal" in self.models:
                # Prepare DF (AlphaEnsemble handles filtering)
                subset = df.loc[idx_normal]
                pred = self.models["normal"].predict_proba(subset)
                scores.loc[idx_normal] = pred
            else:
                logger.warning("Normal model not trained. Defaulting to 0.5 for normal regime samples.")
                scores.loc[idx_normal] = 0.5
                
        # Predict Crisis
        if len(idx_crisis) > 0:
            if "crisis" in self.models:
                subset = df.loc[idx_crisis]
                pred = self.models["crisis"].predict_proba(subset)
                scores.loc[idx_crisis] = pred
            elif "normal" in self.models: # Fallback if crisis model not explicitly trained but normal is
                logger.info("Crisis model not trained, falling back to Normal model for crisis regime samples.")
                subset = df.loc[idx_crisis]
                pred = self.models["normal"].predict_proba(subset)
                scores.loc[idx_crisis] = pred
            else:
                logger.warning("Neither Crisis nor Normal model trained. Defaulting to 0.5 for crisis regime samples.")
                scores.loc[idx_crisis] = 0.5
                
        score_stacking = scores.fillna(0.5) # Fill any remaining NaNs (shouldn't happen if all paths covered)
        
        # Fuse
        # Maybe re-weight dynamically based on regime?
        # User said "Combine predictions if needed".
        # For now, simple weight fusion.
        
        final_score = (
            self.weights["rank_agg"] * score_rank_agg +
            self.weights["weighted"] * score_weighted +
            self.weights["stacking"] * score_stacking
        )
        
        return final_score

    def save(self, path):
         # Logic to pickle self 
         import pickle
         with open(path, "wb") as f:
             pickle.dump(self, f)
             
    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
