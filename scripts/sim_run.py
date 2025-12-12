
import argparse
import random
import time
import json
import logging
from src.sim.event_bus import sim_bus
from src.sim.exchange import exchange_sim
from src.sim.scenario_presets import SCENARIOS

# v41 Sim: CLI Runner
# Orchestrates the simulation run.

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [SIM] - %(message)s")
logger = logging.getLogger("sim_runner")

def run_simulation(scenario_name: str, duration: float, seed: int):
    logger.info(f"🎬 Starting Simulation: {scenario_name} (Seed: {seed})")
    
    # 1. Determinism
    random.seed(seed)
    sim_bus.reset()
    exchange_sim.book = exchange_sim.book.__class__() # Reset Book
    
    # 2. Setup Agents
    factory = SCENARIOS.get(scenario_name)
    if not factory:
        logger.error(f"❌ Unknown scenario: {scenario_name}")
        return
        
    agents = factory()
    for agent in agents:
        agent.start(delay=random.random())
        
    # 3. Run Loop
    logger.info(f"⏳ Running for {duration} sim-seconds...")
    start_real = time.time()
    events = sim_bus.run_until(duration)
    elapsed = time.time() - start_real
    
    # 4. Report
    tps = events / elapsed if elapsed > 0 else 0
    snapshot = exchange_sim.book.get_snapshot()
    
    report = {
        "scenario": scenario_name,
        "duration_sim": duration,
        "events_processed": events,
        "throughput_tps": int(tps),
        "final_snapshot": snapshot
    }
    
    print(json.dumps(report, indent=2))
    logger.info(f"✅ Simulation Complete. Speed: {int(tps)} events/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="flash_crash")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    run_simulation(args.scenario, args.duration, args.seed)
