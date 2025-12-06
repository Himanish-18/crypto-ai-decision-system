import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "features" / "feature_importance.csv"
OUTPUT_PATH = Path("feature_importance_alpha.png")

def main():
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    # Filter for alpha features or top 20
    # Check if alpha features are in the top
    alpha_features = [c for c in df["col_name"] if "alpha" in c]
    print(f"Alpha features found: {alpha_features}")
    
    # Plot top 20
    top_20 = df.head(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_20["col_name"], top_20["feature_importance_vals"])
    plt.gca().invert_yaxis()
    plt.title("Top 20 Feature Importance (including Alphas)")
    plt.xlabel("SHAP Value")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    print(f"Saved plot to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
