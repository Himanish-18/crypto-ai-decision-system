import logging
import os
import sys

sys.path.append(os.getcwd())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("compliance_verifier")


def verify_compliance_modules():
    logger.info("🛡️ Verifying Hedge Fund Compliance Stack...")

    try:
        import src.compliance.trade_journal

        logger.info("✅ src.compliance.trade_journal")

        import src.compliance.exposure_report

        logger.info("✅ src.compliance.exposure_report")

        from src.compliance.compliance_rules import ComplianceRules

        if ComplianceRules.MAX_LEVERAGE > 5.0:
            raise ValueError("MAX_LEVERAGE unsafe!")
        logger.info("✅ src.compliance.compliance_rules (Safety Check Passed)")

        import src.compliance.audit_log

        logger.info("✅ src.compliance.audit_log")

        import src.stats.bootstrap

        logger.info("✅ src.stats.bootstrap")

        import src.backtest.slippage_model_v2

        logger.info("✅ src.backtest.slippage_model_v2")

        import src.data_governance.provenance

        logger.info("✅ src.data_governance")

        logger.info("🎉 All Compliance Modules Imported & Verified.")
        return True
    except Exception as e:
        logger.error(f"❌ Verification Failed: {e}")
        return False


if __name__ == "__main__":
    if not verify_compliance_modules():
        sys.exit(1)
