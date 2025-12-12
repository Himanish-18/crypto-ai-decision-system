
import unittest
from src.sim.event_bus import sim_bus
from src.sim.agents.base_agent import BaseAgent

class MockAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.wakeups = 0
        
    def on_wakeup(self):
        self.wakeups += 1
        
    def on_ticker(self, _): pass
    def on_trade(self, _): pass

class TestAgentIntegration(unittest.TestCase):
    def test_agent_scheduling(self):
        sim_bus.reset()
        agent = MockAgent()
        agent.start()
        
        # Schedule next wakeup
        agent.schedule_wakeup(1.0)
        
        sim_bus.run_until(2.0)
        self.assertGreaterEqual(agent.wakeups, 1)

    def test_agent_orders(self):
        sim_bus.reset()
        agent = MockAgent()
        agent.start()
        
        orders = []
        sim_bus.subscribe("order.new", lambda p: orders.append(p))
        
        agent.place_order("B", 100.0, 1.0)
        sim_bus.run_until(0.1)
        
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['owner_id'], agent.agent_id)
