import asyncio
import logging

from dependencies import session_factory
from services.collector import TrafficCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Запускает периодический сбор данных о трафике."""
    logger.info("Starting traffic data collection")
    while True:
        async with session_factory() as session:
            collector = TrafficCollector(session)
            await collector.collect_data()
        await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
