import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.regime_model import RegimeDetector

logger = logging.getLogger("regime_filter")


class RegimeFilter:
    """
    Market Regime classification wrapper.
    Classes: Bull Trend, Bear Trend, Sideways, High Volatility.
    Wraps existing RegimeDetector but maps to requested 4 categories.
    """

    def __init__(self, model_dir: Path = None):
        if model_dir is None:
            model_dir = Path(__file__).resolve().parents[2] / "data" / "models"
        self.model_path = model_dir / "regime_model.pkl"
        self.detector = RegimeDetector()
        self.labels_path = model_dir.parent / "features" / "regime_labels.parquet"

    def fit_predict_and_save(
        self, df: pd.DataFrame, symbol: str = "btc"
    ) -> pd.DataFrame:
        """Fit detector, predict labels, and save to parquet."""
        logger.info("🛡️ Fitting Regime Detector...")

        # Fit GMM
        self.detector.fit(df, symbol)

        # Predict
        raw_regimes = self.detector.predict(df, symbol)

        # Map raw regimes (Sideways, Trending, HighVol) to requested 4 classes:
        # Bull Trend, Bear Trend, Sideways, High Volatility.
        # "Trending" from GMM is direction-agnostic. We need to split by price trend.

        final_regimes = []
        for i, reg in enumerate(raw_regimes):
            if reg == "Trending":
                # Check Trend Direction (e.g. Price > MA or MACD > 0)
                # We can use simple MA slope or Price vs MA
                close = df[f"{symbol}_close"].iloc[i]
                ma50 = df[f"{symbol}_close"].rolling(50).mean().iloc[i]

                # If MA50 is NaN, fallback
                if np.isnan(ma50):
                    final_regimes.append("Sideways")
                    continue

                if close > ma50:
                    final_regimes.append("Bull Trend")
                else:
                    final_regimes.append("Bear Trend")

            elif reg == "HighVol":
                final_regimes.append("High Volatility")

            else:  # Sideways or Unknown
                final_regimes.append("Sideways")

        # Save Labels
        labels_df = pd.DataFrame(
            {"timestamp": df["timestamp"], "regime": final_regimes}
        )

        # --- REGIME REFINEMENT (Heuristic Overrides) ---
        # 1. Low Liquidity: Volume < 15th Percentile
        # 2. High Volatility: ATR > 95th Percentile OR GMM 'High Volatility'
        # 3. Major Event: Extreme Candle (> 3 sigma move)

        vol = df[f"{symbol}_volume"]
        close = df[f"{symbol}_close"]
        if f"{symbol}_atr_14" in df.columns:
            atr = df[f"{symbol}_atr_14"]
        else:
            # Approx ATR or skip
            atr = (df[f"{symbol}_high"] - df[f"{symbol}_low"]).rolling(14).mean()

        vol_thresh = vol.quantile(0.15)
        atr_thresh = atr.quantile(0.95)

        returns = close.pct_change()
        ret_std = returns.rolling(100).std()
        # Z-score of return > 3
        # Use abs return for magnitude
        is_macro = returns.abs() > (3 * ret_std)

        # Apply Overrides
        # Priority: Major Event > Low Liquidity > High Volatility > Trend/Normal

        refined_regimes = []
        for i, row in enumerate(final_regimes):
            # i corresponds to df index i
            v = vol.iloc[i]
            a = atr.iloc[i]
            m = is_macro.iloc[i]

            # Re-classify
            if m:
                refined_regimes.append("Macro Event")
            elif v < vol_thresh:
                refined_regimes.append("Low Liquidity")
            elif (
                a > atr_thresh or row == "High Volatility"
            ):  # Keep GMM HighVol if reliable
                refined_regimes.append("High Volatility")
            else:
                # Keep original Trend/Normal/Sideways
                # Maybe map Bull/Bear/Sideways to "Normal" for the MultiFactor Model ROUTING?
                # User asked for "Normal regime XGBoost" vs "High Vol".
                # So we can keep detailed labels but ensure Model maps them to 'Normal' bucket.
                # Or just output "Normal" here?
                # Let's keep specific regimes for Risk Engine (which needs Bull/Bear), but Model can group them.
                refined_regimes.append(row)

        labels_df["regime"] = refined_regimes

        logger.info(f"💾 Saving regime labels to {self.labels_path}...")
        self.detector.save(self.model_path)
        labels_df.to_parquet(self.labels_path, index=False)

        return labels_df

    def get_risk_params(self, regime: str) -> dict:
        """Return trading rules based on regime."""
        if regime == "Bull Trend":
            return {
                "stop_loss": 0.03,
                "take_profit": 0.10,
                "position_size": 1.0,
                "entry_threshold": 0.55,
            }
        elif regime == "Bear Trend":
            return {
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "position_size": 0.5,
                "entry_threshold": 0.65,
            }  # Strict
        elif regime == "Sideways":
            return {
                "stop_loss": 0.02,
                "take_profit": 0.03,
                "position_size": 0.8,
                "entry_threshold": 0.60,
            }  # Mean rev
        elif regime == "High Volatility":
            return {
                "stop_loss": 0.05,
                "take_profit": 0.15,
                "position_size": 0.3,
                "entry_threshold": 0.70,
            }  # Safety
        else:
            return {
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "position_size": 0.5,
                "entry_threshold": 0.60,
            }
