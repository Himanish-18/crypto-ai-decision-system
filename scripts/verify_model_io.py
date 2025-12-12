import argparse
import joblib
import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelCheck")

def check_models(patterns):
    logger.info(f"🔍 Checking Models: {patterns}")
    
    # Resolving path
    # Handle wildcard expansion manual if shell didn't? 
    # Python argparse list usually expands if passing * from shell.
    # If passed as string, we verify.
    
    failures = []
    
    for pat in patterns:
        path = Path(pat) 
        if not path.exists():
            # Try glob
            parent = path.parent
            name = path.name
            if "*" in name and parent.exists():
                files = list(parent.glob(name))
            else:
                layout = f"❌ File not found: {pat}"
                logger.error(layout)
                failures.append(layout)
                continue
        else:
            files = [path]
            
        for f in files:
            try:
                logger.info(f"📦 Loading {f}...")
                obj = joblib.load(f)
                logger.info(f"✅ Loaded {f} (Type: {type(obj)})")
            except Exception as e:
                msg = f"❌ FAILED {f}: {e}"
                logger.error(msg)
                failures.append(msg)
                
    if failures:
        sys.exit(1)
    else:
        print("✅ All Models Checked.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Model paths", required=True)
    args = parser.parse_args()
    check_models(args.models)
