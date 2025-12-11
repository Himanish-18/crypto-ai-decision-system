import logging
import statistics
from typing import Dict, List, Optional

logger = logging.getLogger("multi_feed")

class MultiFeedRouter:
    """
    v18 3-Source Data Redundancy.
    Aggregates Funding, OI, and Price from multiple sources.
    Automates Failover.
    """
    def __init__(self):
        self.sources = ["binance", "bybit", "okx"]
        self.active_primary = "binance"
        
    def get_redundant_funding(self) -> float:
        """
        Fetch funding from 3 sources and return Consensus or Median.
        """
        # Mock API calls (In prod, use async fetchers)
        feeds = {
            "binance": 0.0001,
            "bybit": 0.00012,
            "okx": 0.00008
        }
        
        values = list(feeds.values())
        median_val = statistics.median(values)
        
        # Check for deviation (Anomaly Detection)
        for src, val in feeds.items():
            if abs(val - median_val) > 0.0002:
                logger.warning(f"⚠️ Feed Anomaly detected in {src}: {val} vs Median {median_val}")
                
        return median_val
        
    def get_redundant_oi_change(self) -> float:
        """
        Returns consensus OI change %.
        """
        # Stub
        return 0.02
        
    def get_redundant_gvol(self) -> float:
        """
        Returns GVOL from best available source.
        """
        return 0.02
