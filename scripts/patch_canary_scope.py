import sys
import re

CANARY_MAIN = "/tmp/canary_v30_fix/src/main.py"

def patch():
    with open(CANARY_MAIN, "r") as f:
        content = f.read()

    # Identify the Model Loading Block (v40)
    # It starts with "# --- ML MODEL LOADING (v40) ---" and goes until "market_router_v2 = MarketRouterV2()" roughly?
    # Or just indent it?
    # It's currently inside `if __name__ == "__main__":` ?
    # Let's find the block and move it before `def job():` or at least global.
    
    # Block Marker
    start_marker = '# --- ML MODEL LOADING (v40) ---'
    end_marker = '# Update Arbitrator to listen to ML' # Approx
    
    # We need to extract the imports and the loading logic
    # And place them near top of file (after other imports).
    
    # Actually, simpler approach:
    # Just dedent the block so it runs on import?
    # But checking if it's in `if name == main` is hard with regex without context.
    
    # Let's read the file line by line.
    lines = content.splitlines()
    
    new_lines = []
    model_block = []
    in_model_block = False
    
    ml_imports = []
    
    for line in lines:
        if start_marker in line:
            in_model_block = True
            model_block.append(line.strip()) # Dedent
            continue
            
        if in_model_block:
            # Check for end of block heuristic.
            # "execution_quantum = ExecutionQuantum()" seems to be after model load?
            # Or just until we see "market_router_v2"
            if "execution_quantum =" in line:
                in_model_block = False
                new_lines.append(line)
                continue
            
            # If line is import, save to top
            if line.strip().startswith("import ") or line.strip().startswith("from src.ml"):
                ml_imports.append(line.strip())
            else:
                 model_block.append(line.strip())
        else:
            new_lines.append(line)
            
    # Now Insert model_block before `def job():`
    # Warning: `ml_model` needs to be defined before job runs.
    # Insert at top level after imports.
    
    final_lines = []
    inserted = False
    
    for line in new_lines:
        if "def job():" in line and not inserted:
            final_lines.extend(ml_imports)
            final_lines.append("")
            final_lines.extend(model_block)
            final_lines.append("")
            final_lines.append("# Global Model Init Complete")
            final_lines.append(line)
            inserted = True
        else:
            final_lines.append(line)
            
    # Write back
    with open(CANARY_MAIN, "w") as f:
        f.write("\n".join(final_lines))
        
    print("✅ Canary main.py patched.")

if __name__ == "__main__":
    patch()
