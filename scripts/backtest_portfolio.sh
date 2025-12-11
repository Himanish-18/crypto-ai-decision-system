#!/bin/bash
# v23 Portfolio Backtest
echo "🚀 Running v23 Portfolio Backtest..."
export PYTHONPATH=.
python3 scripts/run_portfolio_backtest.py --steps 500
