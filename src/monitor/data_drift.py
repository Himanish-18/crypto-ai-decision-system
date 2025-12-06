import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_drift")

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
REFERENCE_FILE = FEATURES_DIR / "features_1H_advanced.parquet"
DRIFT_REPORT_FILE = DATA_DIR / "analytics" / "drift_report.csv"

class DriftDetector:
    def __init__(self, reference_path: Path = REFERENCE_FILE):
        self.reference_path = reference_path
        self.reference_data = None
        self.numeric_features = []
        
        self.psi_threshold_warning = 0.1
        self.psi_threshold_critical = 0.2
        
    def load_reference(self):
        """Load training data as reference distribution."""
        if not self.reference_path.exists():
            logger.error(f"Reference file not found: {self.reference_path}")
            return
            
        logger.info(f"📥 Loading reference data from {self.reference_path}...")
        self.reference_data = pd.read_parquet(self.reference_path)
        
        # Identify numeric features (exclude timestamp, targets)
        exclude = ["timestamp", "y_direction_up", "btc_ret_fwd_1"]
        self.numeric_features = [c for c in self.reference_data.columns if c not in exclude and np.issubdtype(self.reference_data[c].dtype, np.number)]
        logger.info(f"Tracking {len(self.numeric_features)} numeric features.")

    def calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for a single feature.
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        
        # Define bins based on expected (reference) distribution
        try:
            bins = np.percentile(expected, breakpoints)
        except:
            return 0.0
            
        # Handle duplicate bins
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0
            
        # Calculate counts
        expected_percents = np.histogram(expected, bins)[0] / len(expected)
        actual_percents = np.histogram(actual, bins)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    def check_drift(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """Check drift for all features."""
        if self.reference_data is None:
            self.load_reference()
            
        if self.reference_data is None:
            return pd.DataFrame()
            
        results = []
        
        for feature in self.numeric_features:
            if feature not in live_data.columns:
                continue
                
            psi = self.calculate_psi(self.reference_data[feature], live_data[feature])
            
            status = "OK"
            if psi > self.psi_threshold_critical:
                status = "CRITICAL"
            elif psi > self.psi_threshold_warning:
                status = "WARNING"
                
            results.append({
                "feature": feature,
                "psi": psi,
                "status": status
            })
            
            if status != "OK":
                logger.warning(f"⚠️ Drift detected in {feature}: PSI={psi:.4f} ({status})")
                
        return pd.DataFrame(results)

    def check_regime_shift(self, live_data: pd.DataFrame) -> Dict[str, Any]:
        """Check if regime distribution has shifted significantly."""
        if "regime" not in live_data.columns:
            return {}
            
        # Expected distribution (Approx from training or assumed)
        # Normal: 60%, HighVol: 20%, LowLiq: 10%, Macro: 10%
        # This is a heuristic check.
        
        counts = live_data["regime"].value_counts(normalize=True)
        
        alerts = []
        if counts.get("High Volatility", 0) > 0.40:
            alerts.append("High Volatility Dominance (>40%)")
        
        if counts.get("Low Liquidity", 0) > 0.30:
            alerts.append("Liquidity Crisis (>30%)")
            
        if counts.get("Macro Event", 0) > 0.15:
            alerts.append("Frequent Macro Shocks (>15%)")
            
        if alerts:
            logger.warning(f"🚨 REGIME SHIFT DETECTED: {', '.join(alerts)}")
            return {"status": "DRIFT", "alerts": alerts}
            
        return {"status": "OK"}

def main():
    detector = DriftDetector()
    detector.load_reference()
    
    # Simulate Live Data (Take last 100 rows of reference + some noise to test)
    if detector.reference_data is not None:
        logger.info("🧪 Simulating live data (last 200 rows + noise)...")
        live_data = detector.reference_data.iloc[-200:].copy()
        
        # Introduce drift in one feature
        if "btc_rsi_14" in live_data.columns:
            live_data["btc_rsi_14"] = live_data["btc_rsi_14"] * 1.5 + 10
            logger.info("Simulated drift in btc_rsi_14")
            
        drift_report = detector.check_drift(live_data)
        
        if not drift_report.empty:
            print("\n" + "="*30)
            print(" DRIFT REPORT (Top 5 PSI) ")
            print("="*30)
            print(drift_report.sort_values("psi", ascending=False).head(5).to_string(index=False))
            print("="*30 + "\n")
            
            drift_report.to_csv(DRIFT_REPORT_FILE, index=False)
            logger.info(f"Report saved to {DRIFT_REPORT_FILE}")
        else:
            logger.info("No feature drift calculated.")

        # Check Regime Drift (Simulated)
        if "regime" not in live_data.columns:
            # Add dummy regime for testing
            live_data["regime"] = np.random.choice(["Normal", "High Volatility"], size=len(live_data), p=[0.5, 0.5])
            
        regime_status = detector.check_regime_shift(live_data)
        if regime_status.get("status") == "DRIFT":
            print("\n" + "="*30)
            print(" 🚨 REGIME ALERT ")
            print("="*30)
            print(f"Alerts: {regime_status['alerts']}")
            print("="*30 + "\n")

if __name__ == "__main__":
    main()
