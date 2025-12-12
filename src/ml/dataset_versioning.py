import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger("dataset_versioning")


class DatasetVersionControl:
    """
    Lightweight DVC-like system to track data versions using SHA256 hashes.
    Ensures that every model run is linked to a specific dataset state.
    """

    def __init__(self, data_registry_path: str = "./data/registry.json"):
        self.registry_path = Path(data_registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4)

    def compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def version_dataset(
        self, file_path: str, version_tag: str, description: str = ""
    ) -> str:
        """
        Register a new version of a dataset.
        Returns the hash.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset {file_path} not found.")

        file_hash = self.compute_hash(path)

        entry = {
            "path": str(path),
            "version_tag": version_tag,
            "hash": file_hash,
            "description": description,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Key by tag? Or list?
        # Using composite key: path + version_tag
        key = f"{path.name}:{version_tag}"

        if key in self.registry:
            if self.registry[key]["hash"] != file_hash:
                logger.warning(
                    f"⚠️ Overwriting version {version_tag} with NEW content hash!"
                )

        self.registry[key] = entry
        self._save_registry()
        logger.info(
            f"✅ Registered dataset {path.name} v.{version_tag} (Hash: {file_hash[:8]})"
        )

        return file_hash

    def get_version_info(self, dataset_name: str, version_tag: str) -> Optional[Dict]:
        key = f"{dataset_name}:{version_tag}"
        return self.registry.get(key)

    def verify_dataset(self, file_path: str, expected_hash: str) -> bool:
        """Check if current file matches recorded hash."""
        current_hash = self.compute_hash(Path(file_path))
        return current_hash == expected_hash
