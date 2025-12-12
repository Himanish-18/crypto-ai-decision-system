
import sys
import logging
import random
import time
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harvest_sim")

from src.sim.exchange import SimExchange
from src.sim.event_bus import SimEventBus, SimEvent
from src.sim.agents.liquidity_provider import LiquidityProvider
from src.sim.agents.noise_trader import NoiseTrader
from src.execution.liquidity_harvester import LiquidityHarvester

def run_simulation(duration=600, seed=42):
    random.seed(seed)
    
    # 1. Setup Simulation Environment
    bus = SimEventBus()
    # SimExchange uses global sim_bus internally, but we should align with its API
    # Assuming we modify SimExchange to accept bus or use global.
    # Looking at file, it imports sim_bus global. 
    # But let's check if we can override it or just instantiate.
    exchange = SimExchange() # No args
    exchange.bus = bus # Inject our local bus if needed, or rely on global?
    # Actually, logic relies on global `sim_bus` in exchange.py. 
    # To test safely, we might need to patch it or just use it.
    # For now, just fix the init.
    
    # 2. Add Market Participants (Background Liquidity)
    lp = LiquidityProvider()
    noise = NoiseTrader(activity_freq=0.5)
    
    exchange.bus = bus # Inject bus? (Likely global access in agent too)
    
    # 3. Start Agents (They self-subscribe to bus)
    lp.start(delay=0)
    noise.start(delay=1.0)
    # exchange.add_agent is not needed if they use bus to communicate orders
    
    # 3. Setup Harvester (The Agent Under Test)
    harvester = LiquidityHarvester()
    
    # Mock Data Training for Harvester (so it has a brain)
    from src.execution.fill_probability_model import FillProbabilityModel
    dummy_model = FillProbabilityModel()
    # Train heavily on "easy fills"
    import numpy as np
    X = pd.DataFrame(np.random.rand(100, 5), columns=dummy_model.feature_cols)
    y = pd.Series([1 if x < 0.5 else 0 for x in X['spread_bps']]) # Mock relationship
    dummy_model.train(X, y)
    harvester.fill_model = dummy_model

    logger.info("🚀 Starting Liquidity Harvest Simulation...")
    
    # 4. Simulation Loop
    target_qty = 10.0
    side = "BUY"
    filled_qty = 0.0
    
    start_time = time.time()
    
    step_size = 1.0
    current_sim_time = 0.0
    
    while current_sim_time < duration:
        # Advance Simulation
        bus.run_until(current_sim_time + step_size)
        current_sim_time += step_size
        
        # Harvester Decision every 10s
        if int(current_sim_time) % 10 == 0:
            remaining = target_qty - filled_qty
            if remaining <= 0:
                logger.info("✅ Harvest Complete!")
                break
                
            # Get Market Data Snapshot
            # Stubbing market data from L3 book (would need exchange.get_snapshot())
            # For this test, we construct a dummy snapshot
            market_data = {
                "bid": 100.0 + (random.random() * 0.5),
                "ask": 100.5 + (random.random() * 0.5),
                "volatility": 0.02
            }
            
            # Execute
            decision = harvester.execute_parent_order("BTC-USD", side, remaining, market_data)
            
            # Simulate Outcome
            if decision['action'] == "TAKE":
                # Instant Fill
                fill_amt = decision['qty']
                filled_qty += fill_amt
                logger.info(f"⚡ TAKEN: {fill_amt:.4f} @ {decision['price']:.2f}")
            elif decision['action'] == "POST":
                # Passive placement
                # Check if filled in next 10s (simplified)
                # Using FillProb to "simulate" the fill
                prob = harvester.fill_model.predict_fill_prob({
                    "spread_bps": 5.0, "dist_to_mid": 0.0, "volatility_1m": 0.02, 
                    "depth_skew": 0.0, "trade_flow_imbalance": 0.0
                })
                
                # Roll dice
                if random.random() < prob:
                    fill_amt = decision['qty']
                    filled_qty += fill_amt
                    logger.info(f"💧 PASIVE FILL: {fill_amt:.4f} @ {decision['price']:.2f} (Capture Spread!)")
                else:
                    logger.info(f"⏳ Resting... (Prob {prob:.2f})")
    
    end_time = time.time()
    logger.info(f"🏁 Simulation Ended. Filled: {filled_qty:.4f}/{target_qty} in {end_time - start_time:.2f}s (Simulated {current_sim_time}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_simulation(seed=args.seed)
