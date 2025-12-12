
import subprocess
import sys
import logging

# v40 Infrastructure: CI Pipeline Runner
# Orchestrates Linting, Testing, and Security Audits locally or in CI.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [CI] - %(message)s")
logger = logging.getLogger("ci_runner")

def run_step(name, command):
    logger.info(f"🚀 Running Step: {name}")
    try:
        subprocess.check_call(command, shell=True)
        logger.info(f"✅ {name} Passed.")
        return True
    except subprocess.CalledProcessError:
        logger.error(f"❌ {name} FAILED!")
        return False

def main():
    logger.info("🎬 Starting QuantGrade CI Pipeline...")
    
    steps = [
        ("Linting (Flake8)", "flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics"),
        ("Type Checking (Mypy)", "mypy src/services --ignore-missing-imports"),
        ("Unit Tests", "pytest tests/ --maxfail=1"),
        ("Security Audit (Bandit)", "bandit -r src/ -ll -ii")
    ]
    
    success = True
    for name, cmd in steps:
        if not run_step(name, cmd):
            success = False
            break # Fail Fast
            
    if success:
        logger.info("🏆 CI PIPELINE PASSED! Ready for Deployment.")
        sys.exit(0)
    else:
        logger.error("🚫 CI PIPELINE FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
