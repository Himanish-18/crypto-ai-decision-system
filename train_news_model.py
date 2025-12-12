import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainNewsModel")


def train_bootstrap_model():
    """
    Train a simple news classifier using a keyword-rich bootstrap dataset.
    Labels: 1 (Bull), -1 (Bear), 0 (Neutral)
    """
    data = [
        # Bearish (-1)
        ("Exchange hacked, millions stolen", -1),
        ("Securities and Exchange Commission sues crypto exchange", -1),
        ("Bitcoin crashes below support level", -1),
        ("Regulation ban on mining announced", -1),
        ("Market panic as stablecoin depegs", -1),
        ("Developers abandon project, rug pull suspected", -1),
        ("Interest rate hike causes crypto selloff", -1),
        ("Binance to cease operations in region", -1),
        ("FTX collapse triggers contagion", -1),
        ("Whale dumps large amount of BTC", -1),
        ("Network outage halts transactions", -1),
        ("China bans cryptocurrency transactions", -1),
        # Bullish (1)
        ("Bitcoin hits new all-time high", 1),
        ("ETF approval expected by analysts", 1),
        ("Major bank launches crypto trading desk", 1),
        ("Institutional inflows reach record high", 1),
        ("Partnership with tech giant announced", 1),
        ("Mainnet launch successful", 1),
        ("Defi protocol TVL hits billions", 1),
        ("Country adopts Bitcoin as legal tender", 1),
        ("Google integrates crypto payments", 1),
        ("Bull market confirmed by golden cross", 1),
        ("BlackRock applies for Bitcoin ETF", 1),
        # Neutral (0)
        ("Market consolidates in tight range", 0),
        ("Developer conference scheduled for next month", 0),
        ("Understanding the blockchain consensus mechanism", 0),
        ("Crypto market cap steady at 2 trillion", 0),
        ("New update improves minor bugs", 0),
        ("Weekly analysis of top altcoins", 0),
        ("NFT sales volume report Q3", 0),
        ("Interview with CEO of blockchain startup", 0),
    ]

    df = pd.DataFrame(data, columns=["text", "label"])

    logger.info(f"Training on {len(df)} examples...")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("clf", SVC(kernel="linear", probability=True)),
        ]
    )

    pipeline.fit(df["text"], df["label"])

    out_path = Path("data/models/news_svm.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)

    logger.info(f"✅ Model saved to {out_path}")

    # Quick Test
    test_phrases = ["Fed raises rates", "Tesla buys Bitcoin"]
    preds = pipeline.predict(test_phrases)
    logger.info(f"Test Predictions: {dict(zip(test_phrases, preds))}")


if __name__ == "__main__":
    train_bootstrap_model()
