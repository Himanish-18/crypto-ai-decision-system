import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from src.autonomous.training_loop import TrainingLoop

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - MANAGER - %(message)s")
logger = logging.getLogger("autonomous_brain")

PROJECT_ROOT = Path(__file__).resolve().parent


def run_trading_bot():
    """
    Subprocess for the main trading bot.
    """
    logger.info("🚀 Launching Live Trading Bot (Subprocess)...")
    try:
        # Run main.py
        subprocess.run(["python3", "src/main.py"], check=True)
    except Exception as e:
        logger.error(f"Trading Bot Crashed: {e}")


def run_stream_daemon():
    """
    Subprocess for Data Ingestion Daemon.
    """
    logger.info("🌊 Launching Data Stream Daemon...")
    try:
        subprocess.run(["python3", "src/data/live_stream_daemon.py"], check=True)
    except Exception as e:
        logger.error(f"Stream Daemon Crashed: {e}")


def run_artl_loop():
    """
    ARTL Learning Loop (Labeling + Training).
    """
    from src.data.feature_labeler import FeatureLabeler
    from src.models.self_trainer import SelfTrainer

    labeler = FeatureLabeler()
    trainer = SelfTrainer()

    logger.info("🧠 Initializing ARTL Loop...")

    while True:
        try:
            # 1. Label Data
            labeler.process_daily_buffer()

            # 2. Train
            trainer.incremental_train()

        except Exception as e:
            logger.error(f"ARTL Cycle Error: {e}")

        time.sleep(60)  # Fast loop for Demo


if __name__ == "__main__":
    logger.info("🤖 SYSTEM V14: PORTFOLIO-RL ARTL BRAIN ACTIVATED")

    # 1. Start Stream Daemon (Thread/Subprocess)
    t_stream = threading.Thread(target=run_stream_daemon, daemon=True)
    t_stream.start()

    # 2. Start ARTL Loop (Thread)
    t_artl = threading.Thread(target=run_artl_loop, daemon=True)
    t_artl.start()

    # 3. Start Trading Bot (Blocking)
    while True:
        logger.info("checking bot status...")
        run_trading_bot()
        logger.warning("Bot Process Ended. Restarting in 5s...")
        time.sleep(5)
