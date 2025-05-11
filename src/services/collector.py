import logging
from datetime import UTC, datetime

from datasources.calculated import CalculatedDataSource
from datasources.dataset import DatasetDataSource
from datasources.snmp import SNMPDataSource
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models import Traffic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficCollector:
    """Класс для сбора данных о трафике с использованием различных источников."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_source = self._get_data_source()

    def _get_data_source(self):
        """Возвращает источник данных на основе конфигурации."""
        if settings.DATA_SOURCE == "calculated":
            return CalculatedDataSource()
        elif settings.DATA_SOURCE == "dataset":
            return DatasetDataSource()
        elif settings.DATA_SOURCE == "network":
            return SNMPDataSource()
        raise ValueError(f"Unknown data source: {settings.DATA_SOURCE}")

    async def collect_data(self) -> None:
        """Собирает и сохраняет данные о трафике."""
        try:
            current_time = datetime.now(UTC)
            data = await self.data_source.collect_data(settings.NETWORK_NODES, current_time)

            for entry in data:
                traffic = Traffic(**entry)
                self.db.add(traffic)
                logger.info(
                    f"Добавлены данные для {entry['node_id']}: bandwidth={entry['bandwidth']:.2f}, latency={entry['latency']:.2f}"
                )

            await self.db.commit()
        except Exception as e:
            logger.error(f"Ошибка при сборе данных: {e}")
            raise
