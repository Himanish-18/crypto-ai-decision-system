import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np

logger = logging.getLogger("news_classifier")


class NewsSentimentModel:
    def __init__(self, model_path="data/models/news_svm.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.load()

    def load(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ News Classifier loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load news model: {e}")
        else:
            logger.warning(
                f"⚠️ News Classifier not found at {self.model_path}. Run training script."
            )

    def predict(self, headlines: List[str]) -> float:
        """
        Returns average sentiment score (-1 to 1).
        If multiple headlines, averages the predictions.
        """
        if not self.model or not headlines:
            return 0.0

        try:
            # Predict individual headlines
            # Model output: -1, 0, 1
            preds = self.model.predict(headlines)

            # Simple average
            avg_sentiment = np.mean(preds)
            return float(avg_sentiment)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0
