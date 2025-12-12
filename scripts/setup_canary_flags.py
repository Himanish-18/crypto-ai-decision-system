import yaml
import os

CANARY_CONFIG = "/tmp/canary_v30_fix/config/config.yaml"
CANARY_MAIN = "/tmp/canary_v30_fix/src/main.py"

def update_config():
    with open(CANARY_CONFIG, "r") as f:
        data = yaml.safe_load(f)
        
    if "deployment" not in data:
        data["deployment"] = {}
    if "feature_flags" not in data["deployment"]:
        data["deployment"]["feature_flags"] = {}
        
    # Set to TRUE initially as per instructions "first set them true"
    data["deployment"]["feature_flags"]["disable_ml"] = True
    data["deployment"]["feature_flags"]["disable_dot"] = True
    
    with open(CANARY_CONFIG, "w") as f:
        yaml.dump(data, f)
    print("✅ Canary Config Updated (disable_ml=True, disable_dot=True)")

def patch_main():
    with open(CANARY_MAIN, "r") as f:
        lines = f.readlines()
        
    # We need to:
    # 1. Import yaml (if not present)
    # 2. property load config globally (GLOBAL_CONFIG)
    # 3. Patch job() to check flags
    
    new_lines = []
    yaml_imported = False
    config_loaded = False
    
    # 1. Import YAML
    for line in lines:
        if "import yaml" in line:
            yaml_imported = True
        new_lines.append(line)
        if "import json" in line and not yaml_imported: # heuristic
             new_lines.append("import yaml\n")
             yaml_imported = True
             
    # 2. Parse again to insert Config Load
    # Insert after GLOBAL STATE INIT
    final_lines = []
    for line in new_lines:
        final_lines.append(line)
        if "# --- GLOBAL STATE INIT ---" in line:
             final_lines.append("GLOBAL_CONFIG = {}\n")
             final_lines.append("try:\n")
             final_lines.append('    with open(PROJECT_ROOT / "config" / "config.yaml", "r") as f:\n')
             final_lines.append('        GLOBAL_CONFIG = yaml.safe_load(f)\n')
             final_lines.append("except Exception as e:\n")
             final_lines.append('    logger.warning(f"Config Load Failed: {e}")\n')
             config_loaded = True
             
    # 3. Patch Usage
    # We need to find where `ml_model` is USED.
    # It is used in `if ml_model and TRAINING_FEATURES:` (Line ~295/360)
    # AND `if dot_model:`
    
    patched_lines = []
    
    for line in final_lines:
        # Check for ML usage
        if "if ml_model and TRAINING_FEATURES:" in line:
            # Wrap with flag check
            # We can change the line to:
            # if not GLOBAL_CONFIG.get("deployment", {}).get("feature_flags", {}).get("disable_ml", False) and ml_model and TRAINING_FEATURES:
            indent = line[:line.find("if")]
            patched_lines.append(f'{indent}disable_ml = GLOBAL_CONFIG.get("deployment", {{}}).get("feature_flags", {{}}).get("disable_ml", False)\n')
            patched_lines.append(line.replace("if ", "if not disable_ml and "))
        # Check for DOT usage
        elif "if dot_model:" in line:
            indent = line[:line.find("if")]
            patched_lines.append(f'{indent}disable_dot = GLOBAL_CONFIG.get("deployment", {{}}).get("feature_flags", {{}}).get("disable_dot", False)\n')
            patched_lines.append(line.replace("if ", "if not disable_dot and "))
        else:
            patched_lines.append(line)
            
    with open(CANARY_MAIN, "w") as f:
        f.writelines(patched_lines)
    print("✅ Canary main.py patched with Toggle Checks.")

if __name__ == "__main__":
    update_config()
    patch_main()
