#!/bin/bash

# Load Environment Variables (Ensure this file is secure!)
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Activate Virtual Environment
source venv/bin/activate

# Run the Bot
# Use nohup for background execution if not using systemd
# nohup python -m src.execution.live_loop > live_trading.log 2>&1 &

# For foreground execution (systemd)
python -m src.execution.live_loop
