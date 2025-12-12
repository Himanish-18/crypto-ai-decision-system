
import unittest
from src.sim.scenario_presets import create_flash_crash_scenario
from src.sim.event_bus import sim_bus
from src.sim.exchange import exchange_sim

class TestSimScenarioFlashCrash(unittest.TestCase):
    def test_flash_crash_mechanics(self):
        sim_bus.reset()
        exchange_sim.book = exchange_sim.book.__class__()
        
        agents = create_flash_crash_scenario()
        for agent in agents:
            agent.start()
            
        # Run until t=20 (Crash happens t=10-15)
        sim_bus.run_until(20.0)
        
        snapshot = exchange_sim.book.get_snapshot()
        self.assertTrue(snapshot['best_bid'] < 10000.0, "Price should have crashed")
        # Check if spread widened or book collapsed
        self.assertTrue(snapshot['spread'] >= 0.0)
