#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.guardian.sentinel import Sentinel

def main():
    print("=" * 60)
    print("👁️  v43 AUTONOMOUS SENTINEL STARTING")
    print("=" * 60)
    
    sentinel = Sentinel()
    try:
        sentinel.run_forever()
    except KeyboardInterrupt:
        print("\n🛑 Sentinel stopping...")

if __name__ == "__main__":
    main()
