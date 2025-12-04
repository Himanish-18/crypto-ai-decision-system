import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DASHBOARDS_DIR = Path("dashboards")
DASHBOARDS_DIR.mkdir(exist_ok=True)

def test_plot():
    print("Testing plot generation...")
    plt.figure(figsize=(5, 5))
    sns.barplot(x=[1, 2, 3], y=["A", "B", "C"])
    plt.title("Test Plot")
    save_path = DASHBOARDS_DIR / "test_plot.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    test_plot()
