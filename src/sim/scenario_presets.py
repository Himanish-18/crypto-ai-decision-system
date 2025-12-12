
from typing import List, Callable
import math
from src.sim.event_bus import sim_bus
from src.sim.exchange import exchange_sim
from src.sim.agents.liquidity_provider import LiquidityProvider
from src.sim.agents.noise_trader import NoiseTrader
from src.sim.agents.informed_agent import InformedAgent

# v41 Sim: Scenario Presets
# Factory for configuring agent populations for specific stress tests.

def create_flash_crash_scenario() -> List:
    agents = []
    
    # 1. Normal Liquidity Providers (Withdraw during crash)
    for _ in range(5):
        agents.append(LiquidityProvider(spread_bps=5.0, size_range=(1.0, 5.0)))
        
    # 2. Noise Traders (Background hum)
    for _ in range(10):
        agents.append(NoiseTrader(activity_freq=0.5, size=0.1))
        
    # 3. Crash Agent (Aggressive Selling starts t=10)
    def crash_signal(t):
        if 10.0 <= t <= 15.0:
            return -1.0 # MAX SELL
        return 0.0
        
    agents.append(InformedAgent(signal_function=crash_signal, aggression=10.0))
    
    return agents

def create_liquidity_drought_scenario() -> List:
    agents = []
    # Thin Liquidity (Wide Spreads)
    for _ in range(2):
        agents.append(LiquidityProvider(spread_bps=25.0, size_range=(0.1, 0.5)))
        
    # Normal Noise
    for _ in range(5):
        agents.append(NoiseTrader(activity_freq=0.2, size=0.1))
        
    return agents

SCENARIOS = {
    "flash_crash": create_flash_crash_scenario,
    "liquidity_drought": create_liquidity_drought_scenario
}
