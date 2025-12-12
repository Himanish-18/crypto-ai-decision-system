
import os
import pandas as pd
import logging
import time

# v40 Infrastructure: Data Platform (Storage)
# Abstracts underlying storage (Parquet/Delta/S3) for efficient data access.

class DataStorage:
    def __init__(self, base_path="data/storage"):
        self.base_path = base_path
        self.logger = logging.getLogger("data.platform")
        os.makedirs(base_path, exist_ok=True)

    def write_parquet(self, df: pd.DataFrame, dataset_name: str, partition_by=None):
        """Writes DataFrame to Parquet with optional partitioning."""
        path = os.path.join(self.base_path, f"{dataset_name}.parquet")
        try:
            df.to_parquet(path, index=False, partition_cols=partition_by)
            self.logger.info(f"💾 Written {len(df)} rows to {path}")
        except Exception as e:
            self.logger.error(f"❌ Storage Write Failed: {e}")

    def read_parquet(self, dataset_name: str) -> pd.DataFrame:
        """Reads Parquet dataset."""
        path = os.path.join(self.base_path, f"{dataset_name}.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return pd.DataFrame()

    def version_dataset(self, dataset_name: str):
        """Simulate DVC checkout/commit."""
        self.logger.info(f"📦 DVC: Versioned {dataset_name} at {time.time()}")

storage = DataStorage()
