import yaml
import re

CANARY_MAIN = "/tmp/canary_v30_fix/src/main.py"
CANARY_CONFIG = "/tmp/canary_v30_fix/config/config.yaml"

def update_config_enable():
    with open(CANARY_CONFIG, "r") as f:
        data = yaml.safe_load(f)
    if "deployment" in data and "feature_flags" in data["deployment"]:
        data["deployment"]["feature_flags"]["disable_ml"] = False
        data["deployment"]["feature_flags"]["disable_dot"] = False
    with open(CANARY_CONFIG, "w") as f:
        yaml.dump(data, f)
    print("✅ Config updated: disable_ml=False, disable_dot=False")

def patch_main_safeload():
    with open(CANARY_MAIN, "r") as f:
        lines = f.readlines()
        
    new_lines = []
    
    # 1. Insert safe_load_model function definition
    # We insert it after imports, before globals init or part of globals init.
    # It needs joblib, logger.
    
    safeload_def = [
        "\ndef safe_load_model(path):\n",
        "    try:\n",
        "        if not path.exists():\n",
        "            logger.warning(f'Model missing: {path}')\n",
        "            return None\n",
        "        model = joblib.load(path)\n",
        "        logger.info(f'✅ Safe Load Success: {path.name}')\n",
        "        return model\n",
        "    except Exception as e:\n",
        "        logger.error(f'❌ Model Load Failed {path}: {e}')\n",
        "        return None\n\n"
    ]
    
    inserted_def = False
    
    # We also want to patch the loading block to use safe_load_model
    # The loading block is now at the top level (from previous patch).
    # We look for "ml_model = joblib.load..."
    
    for i, line in enumerate(lines):
        # Insert definition before GLOBAL STATE INIT
        if "# --- GLOBAL STATE INIT ---" in line and not inserted_def:
            new_lines.extend(safeload_def)
            new_lines.append(line)
            inserted_def = True
            continue
            
        # Replace joblib.load with safe_load_model
        if "ml_model = joblib.load(" in line:
            # ml_model = safe_load_model(MODELS_DIR / "multifactor_model_v3.pkl")
            # Extract path arg
            match = re.search(r'joblib\.load\((.+)\)', line)
            if match:
                path_arg = match.group(1)
                new_lines.append(f"        ml_model = safe_load_model({path_arg})\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # 2. Fix Orderflow Delta in calculate_features
    # Search for calculate_features end (before return df) and ensure orderflow_delta is added
    
    final_lines = []
    in_calc = False
    calc_patched = False
    
    for line in new_lines:
        if "def calculate_features" in line:
            in_calc = True
        
        if in_calc and "return df" in line and not calc_patched:
            # Add orderflow_delta
            final_lines.append("    # v40 Fix: Ensure orderflow_delta exists\n")
            final_lines.append("    if 'orderflow_delta' not in df.columns:\n")
            final_lines.append("        # Proxy using volume/close interaction\n")
            final_lines.append("        df['orderflow_delta'] = df['btc_volume'] * (df['btc_close'] - df['btc_open']) / df['btc_close']\n")
            final_lines.append("        df['orderflow_delta'] = df['orderflow_delta'].fillna(0)\n")
            calc_patched = True
            final_lines.append(line)
            in_calc = False # Function ended
        else:
            final_lines.append(line)
            
    with open(CANARY_MAIN, "w") as f:
        f.writelines(final_lines)
    print("✅ Canary main.py patched with safe_load_model and orderflow_delta fix.")

if __name__ == "__main__":
    update_config_enable()
    patch_main_safeload()
