import os
import unittest
from unittest.mock import MagicMock, patch
from src.guardian.monitors.logic_monitor import LogicMonitor
from src.guardian.sentinel import Sentinel
from src.guardian.healer import SentinelHealer
from src.guardian.monitors.base_monitor import Anomaly

class TestSentinelLogic(unittest.TestCase):
    def setUp(self):
        # Create empty file first
        with open("test_predictions.log", "w") as f:
            f.write("")

    def tearDown(self):
        if os.path.exists("test_predictions.log"):
            os.remove("test_predictions.log")

    def test_logic_monitor_detects_lock(self):
        monitor = LogicMonitor("test_predictions.log", lock_limit=10)
        
        # Now write the data
        with open("test_predictions.log", "a") as f:
             for _ in range(12):
                f.write("2025-01-01,Neutral,100.0,Flat,0.0,✅\n")
        
        anomalies = monitor.check()
        self.assertTrue(len(anomalies) > 0)
        self.assertEqual(anomalies[0].type, "NEUTRAL_LOCK")

    @patch("src.guardian.healer.SentinelHealer.relax_noise_threshold")
    @patch("src.guardian.healer.SentinelHealer.restart_service")
    def test_healer_activates_on_lock(self, mock_restart, mock_relax):
        healer = SentinelHealer()
        anomaly = Anomaly("NEUTRAL_LOCK", "CRITICAL", {}, None)
        healer.heal(anomaly)
        
        # Verify it called the fixes
        mock_relax.assert_called_once()
        mock_restart.assert_called_once()

    def test_sentinel_daemon_integration(self):
        sentinel = Sentinel()
        # Mock monitors to avoid real file I/O issues in this specific test
        sentinel.monitors = []
        
        monitor = LogicMonitor("test_predictions.log", lock_limit=10)
        sentinel.register_monitor(monitor)

        # Write data to trigger it
        with open("test_predictions.log", "a") as f:
             for _ in range(12):
                f.write("2025-01-01,Neutral,100.0,Flat,0.0,✅\n")
        
        # Mock Healer
        sentinel.healer.heal = MagicMock()
        
        sentinel.run_once()
        
        # Should detect anomaly and call heal
        sentinel.healer.heal.assert_called()

if __name__ == "__main__":
    unittest.main()
