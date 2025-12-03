#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the main orchestrator
echo "🚀 Starting Crypto AI Decision System..."
python -m src.main
