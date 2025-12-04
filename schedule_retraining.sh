#!/bin/bash

# Path to project
PROJECT_DIR=$(pwd)
PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/src/models/auto_retrain.py"
LOG_FILE="$PROJECT_DIR/data/logs/retrain.log"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/data/logs"

# Cron Expression: Every Sunday at 00:00
CRON_JOB="0 0 * * 0 cd $PROJECT_DIR && $PYTHON_EXEC $SCRIPT_PATH >> $LOG_FILE 2>&1"

# Check if job already exists
(crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH") && echo "Job already exists." && exit 0

# Add job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Retraining job scheduled for every Sunday at 00:00."
echo "Log file: $LOG_FILE"
