import logging
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from enums import NodeType, TrafficType
from models import Node, OptimizationAction

from .base import DataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalculatedDataSource(DataSource):
    """Источник данных, генерирующий метрики на основе расчетов с поддержкой всех типов узлов."""

    def __init__(self, db: AsyncSession | None = None) -> None:
        logger.info("Инициализация CalculatedDataSource")
        self.db = db
        self.node_types = {
            NodeType.fiber: {
                "base_bandwidth": 1000,
                "base_latency": 5,
                "base_packet_loss": 0.1,
                "capacity": 1000,
                "jitter": 2,
                "switch_time": 0.1,
                "switch_loss": 0.5,
            },
            NodeType.gen5: {
                "base_bandwidth": 500,
                "base_latency": 20,
                "base_packet_loss": 1,
                "capacity": 500,
                "jitter": 8,
                "switch_time": 0.2,
                "switch_loss": 1.0,
            },
            NodeType.satellite: {
                "base_bandwidth": 50,
                "base_latency": 600,
                "base_packet_loss": 2,
                "capacity": 100,
                "jitter": 50,
                "switch_time": 0.5,
                "switch_loss": 2.0,
            },
            NodeType.microwave: {
                "base_bandwidth": 200,
                "base_latency": 30,
                "base_packet_loss": 1.5,
                "capacity": 300,
                "jitter": 10,
                "switch_time": 0.3,
                "switch_loss": 1.2,
            },
            NodeType.starlink: {
                "base_bandwidth": 150,
                "base_latency": 80,
                "base_packet_loss": 1.8,
                "capacity": 250,
                "jitter": 15,
                "switch_time": 0.4,
                "switch_loss": 1.5,
            },
            NodeType.hybrid: {
                "base_bandwidth": 600,
                "base_latency": 40,
                "base_packet_loss": 1.2,
                "capacity": 800,
                "jitter": 12,
                "switch_time": 0.25,
                "switch_loss": 1.1,
            },
        }
        self.traffic_types = {
            TrafficType.VOICE: {
                "bandwidth_factor": 0.05,
                "latency_sensitivity": 2.0,
                "jitter_sensitivity": 2.0,
                "packet_loss_sensitivity": 2.0,
            },
            TrafficType.VIDEO: {
                "bandwidth_factor": 0.4,
                "latency_sensitivity": 1.5,
                "jitter_sensitivity": 1.8,
                "packet_loss_sensitivity": 1.7,
            },
            TrafficType.INTERACTIVE: {
                "bandwidth_factor": 0.2,
                "latency_sensitivity": 1.8,
                "jitter_sensitivity": 1.6,
                "packet_loss_sensitivity": 1.4,
            },
            TrafficType.STREAMING: {
                "bandwidth_factor": 0.3,
                "latency_sensitivity": 1.2,
                "jitter_sensitivity": 1.5,
                "packet_loss_sensitivity": 1.3,
            },
            TrafficType.DATA: {
                "bandwidth_factor": 0.15,
                "latency_sensitivity": 1.0,
                "jitter_sensitivity": 1.0,
                "packet_loss_sensitivity": 1.2,
            },
            TrafficType.IOT: {
                "bandwidth_factor": 0.05,
                "latency_sensitivity": 1.1,
                "jitter_sensitivity": 1.1,
                "packet_loss_sensitivity": 1.2,
            },
            TrafficType.SIGNALING: {
                "bandwidth_factor": 0.02,
                "latency_sensitivity": 1.7,
                "jitter_sensitivity": 1.3,
                "packet_loss_sensitivity": 1.8,
            },
        }

        self.total_traffic_demand = settings.TOTAL_TRAFFIC_DEMAND
        self.traffic_distribution = {}
        self.applied_optimizations = {}
        self.last_optimizations_check = None

    async def _get_nodes_from_db(self) -> list[dict]:
        """Получает список узлов из базы данных."""
        if not self.db:
            logger.info("БД недоступна, использую NETWORK_NODES из настроек")
            network_nodes = settings.NETWORK_NODES
            nodes = []
            for node_id in network_nodes:
                node_type = self._detect_node_type(node_id)
                params = self.node_types[node_type]
                nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": node_type,
                        "max_capacity": params["capacity"],
                    },
                )
            logger.info(f"Созданы узлы из настроек: {nodes}")
            return nodes

        query = select(Node).filter(Node.is_active is True)
        result = await self.db.scalars(query)
        nodes = result.all()

        if not nodes:
            logger.warning("В БД не найдены узлы, использую NETWORK_NODES из настроек")
            network_nodes = settings.NETWORK_NODES
            nodes = []
            for node_id in network_nodes:
                node_type = self._detect_node_type(node_id)
                params = self.node_types[node_type]
                nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": node_type,
                        "max_capacity": params["capacity"],
                    },
                )
            logger.info(f"Созданы узлы из настроек: {nodes}")
            return nodes

        nodes_data = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "max_capacity": node.max_capacity,
            }
            for node in nodes
        ]
        logger.info(f"Получены узлы из БД: {nodes_data}")
        return nodes_data

    def _detect_node_type(self, node_id: str) -> str:
        """Определяет тип узла по его имени."""
        node_types = {
            "fiber": NodeType.fiber,
            "sat": NodeType.satellite,
            "5g": NodeType.gen5,
            "micro": NodeType.microwave,
            "star": NodeType.starlink,
            "hybrid": NodeType.hybrid,
        }

        for key, value in node_types.items():
            if key in node_id.lower():
                return value

        return NodeType.gen5

    def _generate_traffic_distribution(self, nodes: list[dict]) -> None:
        """Генерирует распределение трафика по узлам."""
        total_capacity = sum(self.node_types[node["node_type"]]["capacity"] for node in nodes)
        self.traffic_distribution = {
            node["node_id"]: (self.node_types[node["node_type"]]["capacity"] / total_capacity)
            if total_capacity > 0
            else (1.0 / len(nodes))
            for node in nodes
        }

    async def _get_active_optimizations(self) -> list[dict]:
        """Получает активные оптимизации из базы данных."""
        if not self.db:
            return []

        current_time = datetime.now(UTC)
        if (
            self.last_optimizations_check is None
            or (current_time - self.last_optimizations_check).total_seconds() > 300
        ):
            query = select(OptimizationAction).filter(
                OptimizationAction.is_active is True,
                (OptimizationAction.effective_until is None) | (OptimizationAction.effective_until > current_time),
            )
            result = await self.db.scalars(query)
            optimizations = result.all()
            self.applied_optimizations = {
                opt.slug: {
                    "action_type": opt.action_type,
                    "affected_nodes": opt.affected_nodes,
                    "created_at": opt.created_at,
                    "effective_until": opt.effective_until,
                }
                for opt in optimizations
            }

            self.last_optimizations_check = current_time

        return list(self.applied_optimizations.values())

    async def _apply_optimization_effects(self, node_id: str, node_type: str, metrics: dict) -> dict:
        """Применяет эффекты активных оптимизаций к метрикам.

        Args:
            node_id: Идентификатор узла
            node_type: Тип узла
            metrics: Исходные метрики

        Returns:
            Обновленные метрики с учетом оптимизаций
        """
        optimizations = await self._get_active_optimizations()
        if not optimizations:
            return metrics
        updated_metrics = metrics.copy()
        for opt in optimizations:
            if node_id not in opt["affected_nodes"]:
                continue

            action_type = opt["action_type"].lower()
            if "балансировка" in action_type or "load balancing" in action_type:
                reduction_factor = np.random.uniform(0.7, 0.9)
                if metrics["bandwidth"] / metrics["capacity_mbps"] > 0.7:
                    updated_metrics["bandwidth"] *= reduction_factor
            elif "qos" in action_type or "приоритизация" in action_type:
                latency_improvement = np.random.uniform(0.8, 0.95)
                packet_loss_improvement = np.random.uniform(0.6, 0.85)
                updated_metrics["latency"] *= latency_improvement
                updated_metrics["packet_loss"] *= packet_loss_improvement
            elif "переключение" in action_type or "switch" in action_type:
                updated_metrics["switch_time"] *= 0.7
                updated_metrics["switch_packet_loss"] *= 0.7
            elif "пропускная способность" in action_type or "capacity" in action_type:
                capacity_improvement = np.random.uniform(1.05, 1.15)
                updated_metrics["capacity_mbps"] *= capacity_improvement
            elif "маршрутизация" in action_type or "routing" in action_type:
                if node_type == NodeType.hybrid:
                    updated_metrics["latency"] *= 0.85
            elif "шейпинг" in action_type or "shaping" in action_type:
                if updated_metrics["jitter"] > 5:
                    updated_metrics["jitter"] *= 0.8
            elif "отказоустойчивость" in action_type or "failover" in action_type:
                updated_metrics["packet_loss"] *= 0.85

        return updated_metrics

    async def collect_data(self, nodes: list[str], timestamp: datetime) -> list[dict]:
        """Генерирует расчетные данные трафика для указанных узлов."""
        try:
            all_nodes = await self._get_nodes_from_db()
            logger.info(f"Получено узлов из БД: {len(all_nodes)}")

            if not self.traffic_distribution:
                self._generate_traffic_distribution(all_nodes)
                logger.info("Сгенерировано распределение трафика")

            target_nodes = [node for node in all_nodes if node["node_id"] in nodes]
            if not target_nodes:
                logger.warning(f"Не найдены указанные узлы ({nodes}), использую все узлы")
                target_nodes = all_nodes

            logger.info(f"Собираю данные для {len(target_nodes)} узлов")

            result = []
            hour_factor = 0.7 + 0.6 * np.sin(np.pi * (timestamp.hour % 24) / 12)

            for node in target_nodes:
                node_id = node["node_id"]
                node_type = node["node_type"]
                params = self.node_types[node_type]
                node_share = self.traffic_distribution.get(node_id, 1.0 / len(target_nodes))
                noise = np.random.normal(0, 0.1)
                bandwidth = params["base_bandwidth"] * node_share * hour_factor * (1 + noise)
                bandwidth = min(bandwidth, params["capacity"] * 1.2)
                utilization = bandwidth / params["capacity"]
                latency = params["base_latency"] * (1 + utilization * 1.5 + noise * 0.2)
                packet_loss = params["base_packet_loss"] * (1 + utilization * 2 + noise * 0.3)
                jitter = params["jitter"] * (1 + utilization * 1.2 + noise * 0.2)

                if node_type in [NodeType.satellite, NodeType.gen5, NodeType.microwave, NodeType.starlink]:
                    weather_factor = np.random.uniform(0.8, 1.2)
                    latency *= weather_factor
                    packet_loss *= weather_factor
                    signal_strength = -60 - np.random.uniform(0, 40)
                    interference_level = np.random.uniform(0, 15)
                    error_rate = packet_loss / 100 * np.random.uniform(0.8, 1.5)
                else:
                    signal_strength = None
                    interference_level = None
                    error_rate = None

                switch_probability = 0.05
            switched_from = None
            switch_reason = None
            switch_time = 0.0
            switch_packet_loss = 0.0

            if np.random.random() < switch_probability:
                other_nodes = [n for n in target_nodes if n["node_id"] != node_id]
                if other_nodes:
                    source_node = np.random.choice(other_nodes)
                    switched_from = source_node["node_id"]

                    # Разные причины переключения в зависимости от метрик и случайных факторов
                    if utilization > 0.8:
                        switch_reason = "Высокая загрузка канала"
                    elif latency > params["base_latency"] * 2:
                        switch_reason = "Высокая задержка"
                    elif packet_loss > params["base_packet_loss"] * 3:
                        switch_reason = "Высокие потери пакетов"
                    elif jitter > params["jitter"] * 2:
                        switch_reason = "Высокий джиттер"
                    elif (
                        node_type in [NodeType.satellite, NodeType.microwave, NodeType.starlink]
                        and np.random.random() < 0.4
                    ):
                        switch_reason = "Проблемы с сигналом"
                    elif node_type == NodeType.gen5 and np.random.random() < 0.3:
                        switch_reason = "Интерференция сигнала"
                    elif np.random.random() < 0.3:
                        switch_reason = "Плановое переключение"
                    elif np.random.random() < 0.2:
                        switch_reason = "Техническое обслуживание"
                    else:
                        switch_reason = "Оптимизация маршрута"

                    # Время переключения и потери пакетов зависят от типа узла и причины переключения
                    base_switch_time = params["switch_time"]
                    base_packet_loss = params["switch_loss"]

                    # Увеличиваем значения при проблемах
                    if "Высок" in switch_reason or "Проблем" in switch_reason or "Интерференция" in switch_reason:
                        switch_time = base_switch_time * np.random.uniform(1.2, 2.0)
                        switch_packet_loss = base_packet_loss * np.random.uniform(1.5, 2.5)
                    else:
                        switch_time = base_switch_time * np.random.uniform(0.8, 1.2)
                        switch_packet_loss = base_packet_loss * np.random.uniform(0.8, 1.2)

                metrics = {
                    "node_id": node_id,
                    "node_type": node_type,
                    "bandwidth": bandwidth,
                    "latency": latency,
                    "packet_loss": packet_loss,
                    "capacity_mbps": params["capacity"],
                    "jitter": jitter,
                    "switched_from": switched_from,
                    "switch_reason": switch_reason,
                    "switch_time": switch_time,
                    "switch_packet_loss": switch_packet_loss,
                    "signal_strength": signal_strength,
                    "interference_level": interference_level,
                    "error_rate": error_rate,
                    "timestamp": timestamp,
                }

                updated_metrics = await self._apply_optimization_effects(node_id, node_type, metrics)
                result.append(updated_metrics)

            logger.info(f"Сгенерированы данные для {len(result)} узлов")
            return result
        except Exception as e:
            logger.error(f"Ошибка при сборе данных из источника calculated: {e}")
            return []
