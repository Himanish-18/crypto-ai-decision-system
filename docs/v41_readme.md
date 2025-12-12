
# v41 Multi-Agent Simulation Platform

## Overview
The v41 Simulation Platform allows for high-fidelity, deterministic backtesting of high-frequency strategies against a population of simulated agents (Market Makers, Noise Traders, Informed Traders).

## Key Components
- **EventBus**: Deterministic priority queue managing simulation time.
- **OrderBookL3**: Full depth matching engine supporting partial fills and cancellations.
- **Agents**:
    - `LiquidityProvider`: Posts passive limit orders (grids).
    - `NoiseTrader`: Randomly takes liquidity (creates volume).
    - `InformedAgent`: Trades based on simulated "alpha" signals.

## Running Simulations
Use the `sim_run.py` script to execute scenarios.

### 1. Flash Crash
Simulates a liquidity collapse triggered by aggressive selling.
```bash
python scripts/sim_run.py --scenario flash_crash --duration 60 --seed 42
```

### 2. Available Scenarios
- `flash_crash`: Heavy selling + LP withdrawal.
- `liquidity_drought`: Wide spreads and low depth.

## Extending Agents
Create a new class inheriting from `BaseAgent` in `src/sim/agents/`.
Implement `on_wakeup` to place orders and `on_trade` to react to market data.
