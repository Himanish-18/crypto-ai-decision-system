import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger("config_loader")


class ConfigLoader:
    """
    Loads v25 institutional configuration from YAML.
    """

    DEFAULT_PATH = Path("config/v25_config.yaml")

    @staticmethod
    def load_config(path: str = None) -> Dict[str, Any]:
        path = Path(path) if path else ConfigLoader.DEFAULT_PATH

        if not path.exists():
            logger.warning(f"⚠️ Config file {path} not found. Using defaults.")
            return ConfigLoader.get_defaults()

        with open(path, "r") as f:
            try:
                config = yaml.safe_load(f)
                logger.info(f"✅ Configuration loaded from {path}")
                return config
            except yaml.YAMLError as e:
                logger.error(f"❌ Failed to parse config: {e}")
                return ConfigLoader.get_defaults()

    @staticmethod
    def get_defaults() -> Dict[str, Any]:
        return {
            "risk": {"max_drawdown": 0.20, "risk_per_trade": 0.01},
            "execution": {"mode": "paper", "exchange": "binance"},
        }
