import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import shutil

logger = logging.getLogger("model_selector")

class ModelSelector:
    """
    Evaluates Candidate Models against Production Models.
    Promotes if performance metrics improve.
    """
    def __init__(self, models_dir: Path):
        self.prod_dir = models_dir / "prod"
        self.candidate_dir = models_dir / "candidates"
        self.archive_dir = models_dir / "archive"
        
        # Ensure dirs exist
        self.prod_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_and_promote(self, candidate_name: str, verification_data: pd.DataFrame):
        """
        Compare Candidate vs Prod.
        1. Run inference of Prod Model on Data -> Calc Metrics
        2. Run inference of Candidate Model on Data -> Calc Metrics
        3. If Candidate > Prod: Promote.
        """
        logger.info(f"⚖️ Evaluating Candidate: {candidate_name}")
        
        # Load Metrics (Assuming training loop saved metrics.json alongside model)
        # OR run backtest here. For simplicity/speed, we read attached metrics first.
        
        cand_metrics_path = self.candidate_dir / candidate_name / "metrics.json"
        
        # Check Prod Metrics (Load from prod dir or re-calc)
        prod_metrics_path = self.prod_dir / "current_metrics.json"
        
        if not cand_metrics_path.exists():
            logger.error(f"Candidate metrics not found for {candidate_name}")
            return False
            
        with open(cand_metrics_path, 'r') as f:
            cand_metrics = json.load(f)
            
        prod_metrics = None
        if prod_metrics_path.exists():
            with open(prod_metrics_path, 'r') as f:
                prod_metrics = json.load(f)
        
        # Comparison Logic
        should_promote = False
        
        if prod_metrics is None:
            # No prod model exists? Promote immediately.
            logger.info("No production model found. Promoting Initial Candidate.")
            should_promote = True
        else:
            # Compare PF and DD
            pf_gain = cand_metrics.get("profit_factor", 0) - prod_metrics.get("profit_factor", 0)
            dd_red = prod_metrics.get("max_drawdown", 0) - cand_metrics.get("max_drawdown", 0) # e.g. -10 - (-5) = -5 (worse) wait.
            # DD is usually negative. 
            # Prod -10%, Cand -5%.
            # We want Cand > Prod (closer to 0).
            # So Cand(-5) > Prod(-10).
            improved_dd = cand_metrics.get("max_drawdown", -1.0) > prod_metrics.get("max_drawdown", -1.0)
            improved_pf = cand_metrics.get("profit_factor", 0) > prod_metrics.get("profit_factor", 0)
            
            logger.info(f"Comparison: PF({cand_metrics.get('profit_factor'):.2f} vs {prod_metrics.get('profit_factor'):.2f}) | DD({cand_metrics.get('max_drawdown'):.2f} vs {prod_metrics.get('max_drawdown'):.2f})")
            
            if improved_pf and improved_dd:
                should_promote = True
                logger.info("✅ Candidate Superior: Promoting.")
            else:
                 logger.info("❌ Candidate Inferior/Mixed: Rejecting.")
                 
        if should_promote:
            self._promote(candidate_name)
            return True
        else:
            return False

    def _promote(self, candidate_name: str):
        # 1. Archive current Prod
        # Zip or move files
        # 2. Copy Candidate file to Prod
        
        # Implementing a simple file copy strategy
        # Assuming model file is 'model.pkl' inside the folder
        
        src = self.candidate_dir / candidate_name
        dst = self.prod_dir
        
        # Archive current constituents of prod
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / f"prod_{timestamp}"
        archive_path.mkdir(exist_ok=True)
        
        for item in dst.iterdir():
            if item.is_file():
                shutil.copy2(item, archive_path)
        
        # Deplot Candidate
        # Copy all files from candidate folder to prod folder
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, dst)
                
        # Update 'current_metrics.json' in prod to fit the new candidate
        metrics_src = src / "metrics.json"
        if metrics_src.exists():
            shutil.copy2(metrics_src, dst / "current_metrics.json")
            
        logger.info(f"🚀 Promotion Complete! New Model Version: {candidate_name}")
