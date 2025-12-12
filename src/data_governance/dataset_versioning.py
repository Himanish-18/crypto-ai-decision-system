import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("dataset_versioning_gov")


class DataVersionControl:
    """
    Governance layer for dataset versioning.
    Ensures every model input has a unique, traceable ID.
    """

    def __init__(self, registry_file: str = "data/dvc_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry = self._load()

    def _load(self) -> Dict:
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}

    def register_dataset(
        self, file_path: str, description: str, schema_hash: str
    ) -> str:
        """
        Generate Version ID based on Content + Schema.
        """
        # compute file hash
        with open(file_path, "rb") as f:
            content = f.read()
            content_hash = hashlib.sha256(content).hexdigest()

        version_id = f"v1.{content_hash[:8]}"

        entry = {
            "path": file_path,
            "description": description,
            "content_hash": content_hash,
            "schema_hash": schema_hash,
            "created_at": str(logging.Formatter.formatTime),
        }

        self.registry[version_id] = entry
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=4)

        logger.info(f"💾 Registered Dataset {version_id}")
        return version_id
