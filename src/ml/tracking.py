import json
import logging
import os
import random
from typing import Any, Dict, Optional

import mlflow
import numpy as np

logger = logging.getLogger("ml_tracking")


class ExperimentTracker:
    """
    Unified interface for experiment tracking (MLflow/W&B) and reproducibility.
    """

    @staticmethod
    def set_deterministic(seed: int = 42):
        """
        Set global seeds for reproducibility.
        """
        logger.info(f"🔒 Setting global deterministic seed: {seed}")
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass

    def __init__(self, experiment_name: str = "v25_institutional_experiment"):
        self.experiment_name = experiment_name
        # Check if MLflow tracking URI is set, else default
        if not mlflow.get_tracking_uri():
            mlflow.set_tracking_uri("file:./mlruns")

        mlflow.set_experiment(experiment_name)
        self.run = None

    def start_run(self, run_name: Optional[str] = None):
        """Start an MLflow run."""
        self.run = mlflow.start_run(run_name=run_name)
        logger.info(f"🚀 Started MLflow run: {self.run.info.run_id}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str):
        """Log artifact file."""
        mlflow.log_artifact(local_path)

    def end_run(self):
        """End the current run."""
        mlflow.end_run()


class WandBTracker:
    """
    Placeholder for W&B integration.
    """

    def __init__(self, project_name: str):
        self.project_name = project_name

    def init(self, config: Dict):
        # import wandb
        # wandb.init(project=self.project_name, config=config)
        pass
