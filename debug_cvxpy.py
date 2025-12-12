import sys
print(sys.executable)
try:
    import cvxpy
    print(f"CVXPY Version: {cvxpy.__version__}")
except Exception as e:
    print(f"Import Failed: {e}")
