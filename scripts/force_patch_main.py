import sys
import os

CANARY_MAIN = "/tmp/canary_v30_fix/src/main.py"

def patch():
    with open(CANARY_MAIN, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    globals_inserted = False
    
    # We want to insert BEFORE "def calculate_features" or "def "
    # But checking for LOG_DIR might be safer, or just before the first function.
    
    for i, line in enumerate(lines):
        # Clean up previous bad patch if present?
        # The bad patch inserted lines that broke syntax.
        # We should start from a clean state or remove the bad lines.
        # Detection: "# --- GLOBAL STATE INIT ---"
        if "# --- GLOBAL STATE INIT ---" in line:
            # Skip this and subsequent lines until we hit valid code again?
            # Or assume we are re-patching a clean file?
            # User said "Copy repo into /tmp/canary" manually or via tool. 
            # If I run this on the ALREADY BROKEN file, I need to clean it.
            # Only skipping lines won't fix the fact that "from ... (" is hanging.
            pass 
        
        # Better: Re-copy main.py from source to canary first to reset.
        pass

    # Actually, let's just reset the file first in the script.
    # We can read source main.py
    SOURCE_MAIN = "src/main.py"
    with open(SOURCE_MAIN, "r") as f:
        source_lines = f.readlines()
        
    final_lines = []
    inserted = False
    
    for line in source_lines:
        if "def calculate_features" in line and not inserted:
            final_lines.append("\n# --- GLOBAL STATE INIT ---\n")
            final_lines.append("ml_model = None\n")
            final_lines.append("dot_model = None\n")
            final_lines.append("TRAINING_FEATURES = []\n")
            final_lines.append("EMA_RET = None\n")
            final_lines.append("PRED_QUEUE = collections.deque(maxlen=5)\n")
            final_lines.append("LAST_PRED_DIR = 'Neutral'\n")
            final_lines.append("LAST_PRED_SCORE = 0.0\n")
            final_lines.append("LAST_PRED_PRICE = 0.0\n")
            
            final_lines.append("\n# --- ML MODEL LOADING (Moved to Global) ---\n")
            final_lines.append("try:\n")
            final_lines.append('    if (MODELS_DIR / "multifactor_model_v3.pkl").exists():\n')
            final_lines.append('        ml_model = joblib.load(MODELS_DIR / "multifactor_model_v3.pkl")\n')
            final_lines.append('        with open(MODELS_DIR / "training_features.json", "r") as f:\n')
            final_lines.append('            TRAINING_FEATURES = json.load(f)\n')
            final_lines.append('        logger.info("🧠 Multifactor Model v3 Loaded.")\n')
            final_lines.append('    elif (MODELS_DIR / "multifactor_model_v2.pkl").exists():\n')
            final_lines.append('        ml_model = joblib.load(MODELS_DIR / "multifactor_model_v2.pkl")\n')
            final_lines.append('        logger.warning("⚠️ Using Fallback v2 Model.")\n')
            final_lines.append('except Exception as e:\n')
            final_lines.append('    logger.error(f"Failed to load ML Model: {e}")\n')
            final_lines.append("\n")
            
            inserted = True
            
        final_lines.append(line)
        
    with open(CANARY_MAIN, "w") as f:
        f.writelines(final_lines)
        
    print("✅ Canary main.py RESET and FORCE-PATCHED.")

if __name__ == "__main__":
    patch()
