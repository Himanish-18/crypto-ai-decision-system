import asyncio
import logging
import sys

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.engine.event_bus import EventBus
from src.services.data_stream import DataStreamService
from src.services.execution_gateway import ExecutionGateway
from src.services.portfolio_engine import PortfolioEngine
from src.services.risk_veto import RiskVetoService
from src.services.strategy_core import StrategyCoreService


async def main():
    bus = EventBus()

    # Instantiate Services
    data_svc = DataStreamService(bus)
    strategy_svc = StrategyCoreService(bus)
    risk_svc = RiskVetoService(bus)
    exec_svc = ExecutionGateway(bus)
    portfolio_svc = PortfolioEngine(bus)

    # Start Data Stream
    await data_svc.start()

    # Run Bus
    try:
        await bus.run()
    except KeyboardInterrupt:
        logging.getLogger("main_service").info("Shutdown requested")
    finally:
        data_svc.stop()
        bus.stop()


if __name__ == "__main__":
    asyncio.run(main())
