import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DataSource:
    name: str  # e.g. "Binance", "CoinGecko"
    type: str  # "Websocket", "REST", "CSV"
    uri: str
    fetch_frequency: str  # "1s", "1h"


class ProvenanceTracker:
    """
    Tracks lineage of data points.
    """

    def __init__(self):
        self.registry: Dict[str, DataSource] = {}

    def register_source(self, source: DataSource):
        self.registry[source.name] = source

    def generate_lineage_metadata(self, source_name: str) -> Dict[str, str]:
        source = self.registry.get(source_name)
        if not source:
            return {"provenance": "unknown"}

        return {
            "source": source.name,
            "ingest_time": str(time.time()),
            "origin_uri": source.uri,
        }
