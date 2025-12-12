import json
import logging
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("system_verify")


def run_test(command, version_name):
    logger.info(f"🔵 Running Verification for {version_name}...")
    start = time.time()
    try:
        # Capture Output
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        duration = time.time() - start

        if result.returncode == 0:
            logger.info(f"✅ {version_name} PASSED ({duration:.2f}s)")
            return {
                "version": version_name,
                "status": "PASS",
                "duration": duration,
                "output": result.stdout,
            }
        else:
            logger.error(f"❌ {version_name} FAILED ({duration:.2f}s)")
            logger.error(result.stderr)
            return {
                "version": version_name,
                "status": "FAIL",
                "duration": duration,
                "error": result.stderr,
            }
    except Exception as e:
        logger.error(f"❌ {version_name} CRASH: {e}")
        return {"version": version_name, "status": "CRASH", "error": str(e)}


def main():
    logger.info("🚀 BEGINNING INSTITUTIONAL UPGRADE VERIFICATION PIPELINE")

    results = []

    # v19: Quantum / Arbitrator
    # Assuming quantum_testbench.py exists and runs with unittest
    results.append(
        run_test("python3 tests/quantum_testbench.py", "v19_Quantum_Stacker")
    )

    # v20: Risk Engine
    results.append(run_test("python3 tests/test_portfolio_risk.py", "v20_Risk_Engine"))

    # v21: Regime Detection
    results.append(run_test("python3 tests/test_regime_v21.py", "v21_Regime_Detection"))

    # v22: Execution Logic
    results.append(
        run_test("python3 tests/test_execution_v22.py", "v22_Execution_Logic")
    )

    # v23: Portfolio Opt
    results.append(
        run_test("python3 tests/test_portfolio_env.py", "v23_Portfolio_RL_Unit")
    )
    results.append(
        run_test("bash scripts/backtest_portfolio.sh", "v23_Portfolio_RL_Backtest")
    )

    # Summary
    logger.info("\n📊 CONSOLIDATED REPORT CARD 📊")
    all_pass = True
    for r in results:
        status_icon = "✅" if r["status"] == "PASS" else "❌"
        print(
            f"{status_icon} {r['version']}: {r['status']} ({r.get('duration',0):.2f}s)"
        )
        if r["status"] != "PASS":
            all_pass = False

    if all_pass:
        print("\n🏆 ALL SYSTEMS VERIFIED. READY FOR DEPLOYMENT.")
        # Create JSON for report generation
        with open("upgrade_results.json", "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(0)
    else:
        print("\n⚠️ DEPLOYMENT BLOCKED. FAILURES DETECTED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
