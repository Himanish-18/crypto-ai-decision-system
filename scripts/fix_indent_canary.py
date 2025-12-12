CANARY_MAIN = "/tmp/canary_v30_fix/src/main.py"

def fix():
    with open(CANARY_MAIN, "r") as f:
        lines = f.readlines()
        
    final_lines = []
    
    # We look for the specific lines we inserted and didn't indent enough.
    # Pattern: "ml_model = safe_load_model(" preceded by "if ... exists():"
    
    for i, line in enumerate(lines):
        if "ml_model = safe_load_model(" in line:
            # Check context
            if i > 0 and "exists():" in lines[i-1]:
                # This needs to be indented more than prev line
                prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip())
                curr_indent = len(line) - len(line.lstrip())
                if curr_indent <= prev_indent:
                    # Add 4 spaces
                    final_lines.append("    " + line)
                    print(f"Fixed indent at line {i+1}")
                    continue
        
        # Also clean up the double logic - my previous patch might have left artifacts?
        # The sed output showed:
        # if ... exists():
        # ml_model = ...
        # logger.info...
        # else:
        # ...
        # ml_model = safe_load_model(...v2...)
        #
        # Use simpler logic: Just ensuring 12 spaces if it looks like body.
        final_lines.append(line)
        
    with open(CANARY_MAIN, "w") as f:
        f.writelines(final_lines)
    print("✅ Canary main.py indentation fixed.")

if __name__ == "__main__":
    fix()
