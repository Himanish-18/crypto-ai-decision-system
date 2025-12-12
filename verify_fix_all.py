import importlib
import logging
import os
import sys

sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_all_verifier")


def verify_modules():
    logger.info("🛠️ Verifying Fix-All Patch Modules...")

    modules_to_check = [
        "src.ml.validation.walk_forward",
        "src.ml.validation.time_splitter",
        "src.ml.validation.regime_validation",
        "src.analytics.performance",
        "src.ops.observability.drift_detector",
        "src.ops.observability.latency_monitor",
        "src.risk.institutional_v25.pca_model",
        "src.backtest.slippage_model_v2",
        "src.backtest.market_impact",
    ]

    failed = []

    for mod_name in modules_to_check:
        try:
            importlib.import_module(mod_name)
            logger.info(f"✅ {mod_name}")
        except ImportError as e:
            logger.error(f"❌ {mod_name}: {e}")
            failed.append(mod_name)
        except Exception as e:
            logger.error(f"❌ {mod_name} (Runtime Error): {e}")
            failed.append(mod_name)

    if failed:
        logger.error(f"Failed modules: {failed}")
        return False

    logger.info("🎉 All Patch Modules Verified.")
    return True


if __name__ == "__main__":
    if not verify_modules():
        sys.exit(1)
