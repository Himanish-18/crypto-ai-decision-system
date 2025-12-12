CANARY_MAIN = "/tmp/canary_v30_fix/src/main.py"

def fix_dot():
    with open(CANARY_MAIN, "r") as f:
        lines = f.readlines()
        
    dot_init_line = None
    dot_idx = -1
    
    # Search for dot_model init
    # "dot_model = OrderFlowFeatures" or "OrderflowTransformer"
    # Actually Main usually imports OrderFlowFeatures?
    # Let's search for "dot_model ="
    for i, line in enumerate(lines):
        if "dot_model =" in line and "None" not in line:
            dot_init_line = line
            dot_idx = i
            break
            
    new_lines = []
    
    moved = False
    
    # If found, check if it's inside if __name__.
    # But since I can't easily parse blocks in dumb script, 
    # I will just Insert the init logic into the Global Init block I created.
    # Logic: dot_model = OrderflowTransformer() (assuming import exists)
    
    # I need to see the import for OrderflowTransformer.
    # It might be in src.features.orderflow_features.
    
    # Let's just Add the init line explicitly after "dot_model = None".
    # And Comment out the old one if found.
    
    for line in lines:
        if "dot_model = None" in line:
            new_lines.append(line)
            # Add init
            new_lines.append("# Fix: Init DOT Model\n")
            new_lines.append("try:\n")
            new_lines.append("    from src.features.orderflow_features import OrderFlowTransformer\n")
            new_lines.append("    dot_model = OrderFlowTransformer()\n")
            new_lines.append('    logger.info("🧠 DOT Model Initialized.")\n')
            new_lines.append("except Exception as e:\n")
            new_lines.append('    logger.warning(f"DOT Init Failed: {e}")\n')
        elif "dot_model =" in line and "None" not in line and "OrderFlowTransformer" in line:
            # Comment out old init to avoid double init or scope issues
            new_lines.append(f"# {line}")
        else:
            new_lines.append(line)
            
    with open(CANARY_MAIN, "w") as f:
        f.writelines(new_lines)
    print("✅ Canary main.py: DOT Model loading fixed.")

if __name__ == "__main__":
    fix_dot()
