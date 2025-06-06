import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from datasources.base import DataSource
from datasources.calculated import CalculatedDataSource
from datasources.dataset import DatasetDataSource
from datasources.snmp import SNMPDataSource
from models import NetworkLink, Node, Traffic
from services.optimizer import TrafficOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficCollector:
    """Класс для сбора данных о трафике с использованием различных источников в гибридной сети."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.data_sources = self._initialize_data_sources()
        self.primary_source = settings.DATA_SOURCE
        self.collection_interval = 10
        self.last_optimization_update = None

    def _initialize_data_sources(self) -> dict[str, DataSource]:
        """Инициализирует только выбранный в настройках источник данных с передачей сессии базы данных.

        Returns:
            Словарь с инициализированным источником данных.
        """
        source_name = settings.DATA_SOURCE
        if source_name == "calculated":
            return {"calculated": CalculatedDataSource(db=self.db)}
        elif source_name == "dataset":
            return {"dataset": DatasetDataSource(db=self.db)}
        elif source_name == "network":
            return {"network": SNMPDataSource(db=self.db)}
        else:
            logger.warning(f"Неизвестный источник данных: {source_name}, используем calculated по умолчанию")
            return {"calculated": CalculatedDataSource(db=self.db)}

    def _get_data_source(self, source_name: str | None = None) -> DataSource:
        """Возвращает источник данных на основе конфигурации или переданного имени.

        Args:
            source_name: Опциональное имя источника данных.

        Returns:
            Источник данных.

        Raises:
            ValueError: Если источник данных не найден.
        """
        source = source_name or self.primary_source
        if source in self.data_sources:
            return self.data_sources[source]
        raise ValueError(f"Неизвестный источник данных: {source}")

    async def collect_data(self) -> None:
        """Собирает и сохраняет данные о трафике из настроенного источника."""
        try:
            current_time = datetime.now(UTC)

            if (
                self.last_optimization_update is None
                or (current_time - self.last_optimization_update).total_seconds() > 21600
            ):
                try:
                    optimizer = TrafficOptimizer(self.db)
                    await optimizer.update_optimization_statuses()
                    self.last_optimization_update = current_time
                    logger.info("Обновлены статусы оптимизаций")
                except Exception as e:
                    logger.error(f"Ошибка при обновлении статусов оптимизаций: {e}")

            nodes = await self._get_active_nodes()

            if not nodes:
                logger.warning("Не найдены активные узлы в базе данных")
                return

            node_ids = [node["node_id"] for node in nodes]

            # Используем только выбранный источник данных
            all_data = await self._collect_from_source(self.primary_source, node_ids, current_time)
            
            if not all_data:
                logger.warning(f"Не удалось получить данные из источника {self.primary_source}")
                return

            for entry in all_data:
                entry.pop("data_source", None)
                traffic = Traffic(**entry)
                self.db.add(traffic)
                logger.info(
                    f"Добавлены данные для {entry['node_id']} ({entry['node_type']}): "
                    f"bandwidth={entry['bandwidth']:.2f}, latency={entry['latency']:.2f}, "
                    f"packet_loss={entry['packet_loss']:.2f}",
                )

            await self._generate_network_topology(all_data)

            await self.db.commit()
            logger.info(f"Сохранены данные для {len(all_data)} узлов")

        except Exception as e:
            logger.error(f"Ошибка при сборе данных: {e}")
            await self.db.rollback()
            raise

    async def _get_active_nodes(self) -> list[dict[str, Any]]:
        """Получает список всех активных узлов из базы данных.

        Returns:
            Список словарей с информацией об узлах.
        """
        try:
            query = select(Node).filter(Node.is_active == True)  # Исправлена ошибка: is True -> == True
            result = await self.db.scalars(query)
            nodes = result.all()

            if not nodes and hasattr(settings, "NETWORK_NODES"):
                return [{"node_id": node, "node_type": self._detect_node_type(node)} for node in settings.NETWORK_NODES]

            return [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                }
                for node in nodes
            ]
        except Exception as e:
            logger.error(f"Ошибка при получении списка узлов: {e}")

            if hasattr(settings, "NETWORK_NODES"):
                return [{"node_id": node, "node_type": self._detect_node_type(node)} for node in settings.NETWORK_NODES]
            return []

    def _detect_node_type(self, node_id: str) -> str:
        """Определяет тип узла по его имени (для обратной совместимости).

        Args:
            node_id: Идентификатор узла.

        Returns:
            Тип узла.
        """
        node_types = {
            "fiber": "fiber",
            "sat": "satellite",
            "5g": "5G",
            "micro": "microwave",
            "star": "starlink",
            "hybrid": "hybrid",
        }

        for key, value in node_types.items():
            if key in node_id.lower():
                return value

        return "5G"

    async def _collect_from_source(
        self,
        source_name: str,
        node_ids: list[str],
        timestamp: datetime,
    ) -> list[dict[str, Any]]:
        """Собирает данные из указанного источника.

        Args:
            source_name: Имя источника данных.
            node_ids: Список идентификаторов узлов для сбора данных.
            timestamp: Временная метка.

        Returns:
            Собранные данные о трафике.
        """
        try:
            logger.info(f"Получаю данные из источника: {source_name} для узлов: {node_ids}")
            data_source = self._get_data_source(source_name)
            logger.info(f"Источник данных получен: {type(data_source).__name__}")
            data = await data_source.collect_data(node_ids, timestamp)
            logger.info(f"Получено данных из источника {source_name}: {len(data)}")

            if data:
                for entry in data:
                    entry["data_source"] = source_name

                    if "jitter" not in entry:
                        entry["jitter"] = 0.0

                    if entry["node_type"] in ["5G", "satellite", "starlink", "microwave"]:
                        for field in ["signal_strength", "interference_level", "error_rate"]:
                            if field not in entry:
                                entry[field] = None

            return data
        except Exception as e:
            logger.error(f"Ошибка при сборе данных из источника {source_name}: {e}")
            return []

    async def _generate_network_topology(self, traffic_data: list[dict[str, Any]]) -> None:
        """Генерирует и обновляет информацию о топологии сети.

        Args:
            traffic_data: Собранные данные о трафике.
        """
        for entry in traffic_data:
            node_id = entry["node_id"]
            node_type = entry["node_type"]

            # Используем параметризованный запрос вместо строковой интерполяции
            node_query = select(Node).filter(Node.node_id == node_id)
            existing_node = await self.db.scalar(node_query)

            if not existing_node:
                node = Node(
                    node_id=node_id,
                    node_type=node_type,
                    description=f"{node_type.capitalize()} node - {node_id}",
                    max_capacity=entry["capacity_mbps"],
                    is_active=True,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                self.db.add(node)
                logger.info(f"Создан новый узел: {node_id} ({node_type})")

        if len(traffic_data) > 1:
            node_types = {}
            for entry in traffic_data:
                node_type = entry["node_type"]
                if node_type not in node_types:
                    node_types[node_type] = []
                node_types[node_type].append(entry)

            for node_type, nodes in node_types.items():
                if len(nodes) > 1:
                    for i, source in enumerate(nodes):
                        for j, target in enumerate(nodes):
                            if i != j:
                                # Используем параметризованный запрос вместо строковой интерполяции
                                link_query = select(NetworkLink).filter(
                                    NetworkLink.source_node == source['node_id'],
                                    NetworkLink.target_node == target['node_id']
                                )
                                existing_link = await self.db.scalar(link_query)

                                if not existing_link:
                                    link = NetworkLink(
                                        source_node=source["node_id"],
                                        target_node=target["node_id"],
                                        bandwidth=min(source["bandwidth"], target["bandwidth"]),
                                        latency=(source["latency"] + target["latency"]) / 2,
                                        is_active=True,
                                        link_type=f"{node_type}-{node_type}",
                                        weight=1.0,
                                        created_at=datetime.now(UTC),
                                        updated_at=datetime.now(UTC),
                                    )
                                    self.db.add(link)

            if len(node_types) > 1:
                representative_nodes = {}
                for node_type, nodes in node_types.items():
                    best_node = min(nodes, key=lambda n: n["latency"] / max(n["bandwidth"], 1))
                    representative_nodes[node_type] = best_node

                node_types_list = list(representative_nodes.keys())
                for i, type1 in enumerate(node_types_list):
                    for j, type2 in enumerate(node_types_list[i + 1 :], i + 1):
                        source = representative_nodes[type1]
                        target = representative_nodes[type2]

                        # Используем параметризованный запрос вместо строковой интерполяции
                        link_query = select(NetworkLink).filter(
                            ((NetworkLink.source_node == source['node_id']) & (NetworkLink.target_node == target['node_id'])) |
                            ((NetworkLink.source_node == target['node_id']) & (NetworkLink.target_node == source['node_id']))
                        )
                        existing_link = await self.db.scalar(link_query)

                        if not existing_link:
                            hybrid_link = NetworkLink(
                                source_node=source["node_id"],
                                target_node=target["node_id"],
                                bandwidth=min(source["bandwidth"], target["bandwidth"]) * 0.8,
                                latency=(source["latency"] + target["latency"]) * 0.6,
                                is_active=True,
                                link_type=f"{type1}-{type2}",
                                weight=1.5,
                                created_at=datetime.now(UTC),
                                updated_at=datetime.now(UTC),
                            )
                            self.db.add(hybrid_link)
                            logger.info(
                                f"Создано гибридное соединение: {source['node_id']} ({type1}) -> "
                                f"{target['node_id']} ({type2})",
                            )
