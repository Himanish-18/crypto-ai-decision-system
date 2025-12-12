
import argparse
import logging
from src.rl.meta_trainer import MetaTrainer

# v42 Meta-RL: Training Script
# Runs the MAML training loop.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [META] - %(message)s")
logger = logging.getLogger("meta_train")

def train(iterations, tasks):
    logger.info(f"🚀 Starting Meta-Training (MAML) for {iterations} steps...")
    
    # Obs Dim 8, Act Dim 2 (Stub dimensions)
    trainer = MetaTrainer(obs_dim=8, act_dim=2, tasks=tasks)
    
    for i in range(iterations):
        loss = trainer.meta_update(meta_batch_size=4)
        
        if i % 10 == 0:
            logger.info(f"Step {i}/{iterations} | Meta-Loss: {loss:.4f}")
            
    logger.info("✅ Meta-Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-steps", type=int, default=100)
    parser.add_argument("--tasks", type=int, default=4) # dummy arg for batch size context
    args = parser.parse_args()
    
    train(args.meta_steps, None)
