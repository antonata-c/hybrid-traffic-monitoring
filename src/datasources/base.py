from abc import ABC, abstractmethod
from datetime import datetime


class DataSource(ABC):
    """Абстрактный интерфейс для источников данных."""

    @abstractmethod
    async def collect_data(self, nodes: list[str], timestamp: datetime) -> list[dict]:
        """Собирает данные о трафике для указанных узлов.

        Args:
            nodes: Список узлов сети.
            timestamp: Время сбора данных.

        Returns:
            Список словарей с метриками трафика.
        """
        pass
