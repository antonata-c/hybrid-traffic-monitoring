import asyncio
import logging

from dependencies import session_factory
from services.traffic import collect_traffic_data

logging.getLogger().setLevel(logging.INFO)

async def main():
    logging.info("Starting traffic data collection")
    data_index = 0
    while True:
        async with session_factory() as session:
            await collect_traffic_data(session, data_index)
            data_index += 1
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())