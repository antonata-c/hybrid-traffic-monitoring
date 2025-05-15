import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from enums import NodeType, TrafficType
from models import Node, OptimizationAction

from .base import DataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDataSource(DataSource):
    """Источник данных, использующий датасет Abilene с симуляцией переключений и разнообразной нагрузкой."""

    def __init__(self, db: AsyncSession | None = None) -> None:
        self.db = db
        self.node_types = {
            NodeType.fiber: {
                "base_bandwidth": 1000,
                "base_latency": 5,
                "base_packet_loss": 0.1,
                "capacity": 1000,
                "switch_time": 0.1,
                "switch_loss": 0.5,
                "jitter": 2,
            },
            NodeType.satellite: {
                "base_bandwidth": 50,
                "base_latency": 600,
                "base_packet_loss": 2,
                "capacity": 100,
                "switch_time": 0.5,
                "switch_loss": 2.0,
                "jitter": 50,
            },
            NodeType.gen5: {
                "base_bandwidth": 500,
                "base_latency": 20,
                "base_packet_loss": 1,
                "capacity": 500,
                "switch_time": 0.2,
                "switch_loss": 1.0,
                "jitter": 8,
            },
            NodeType.microwave: {
                "base_bandwidth": 200,
                "base_latency": 30,
                "base_packet_loss": 1.5,
                "capacity": 300,
                "switch_time": 0.3,
                "switch_loss": 1.2,
                "jitter": 10,
            },
            NodeType.starlink: {
                "base_bandwidth": 150,
                "base_latency": 80,
                "base_packet_loss": 1.8,
                "capacity": 250,
                "switch_time": 0.4,
                "switch_loss": 1.5,
                "jitter": 15,
            },
            NodeType.hybrid: {
                "base_bandwidth": 600,
                "base_latency": 40,
                "base_packet_loss": 1.2,
                "capacity": 800,
                "switch_time": 0.25,
                "switch_loss": 1.1,
                "jitter": 12,
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

        try:
            self.data = pd.read_csv("../data/abilene_traffic.csv")
            logger.info("Загружен датасет Abilene")
        except FileNotFoundError:
            logger.warning("Датасет Abilene не найден, используется синтетический датасет")
            self.data = pd.DataFrame(
                {
                    "timestamp": [datetime.now(UTC) - timedelta(minutes=i) for i in range(1000)],
                    "router_id": [f"router_{i % 12}" for i in range(1000)],
                    "traffic_mbps": [10.0 * (i % 100 + 1) for i in range(1000)],
                },
            )

        self.node_to_router = {}
        self.index = 0
        self.switch_history = {}
        self.node_distances = {}
        self.node_status = {}
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

    def _initialize_node_mappings(self, nodes: list[dict]) -> None:
        """Инициализирует отображения и статусы узлов."""
        self.node_to_router = {node["node_id"]: f"router_{i % 12}" for i, node in enumerate(nodes)}

        self.node_distances = {node["node_id"]: np.random.randint(10, 1000) for node in nodes}

        self.node_status = {node["node_id"]: {"available": True, "load_factor": 1.0} for node in nodes}

    def _simulate_network_events(self, node_id: str, node_type: str) -> dict:
        """Симулирует сетевые события: сбои, пики нагрузки, ухудшение метрик."""
        status = self.node_status[node_id]

        failure_probabilities = {
            NodeType.fiber: 0.005,
            NodeType.gen5: 0.01,
            NodeType.satellite: 0.02,
            NodeType.microwave: 0.015,
            NodeType.starlink: 0.012,
            NodeType.hybrid: 0.008,
        }
        failure_prob = failure_probabilities.get(node_type, 0.01)

        if np.random.random() < failure_prob:
            status["available"] = False
            status["event"] = "Сбой сети"
        else:
            status["available"] = True
            status["event"] = None

        if np.random.random() < 0.1:
            status["load_factor"] = np.random.uniform(1.5, 3.0)
            status["event"] = "Пик нагрузки" if not status["event"] else status["event"]
        else:
            status["load_factor"] = np.random.uniform(0.8, 1.2)

        return status

    def _calculate_route_score(
        self,
        node_id: str,
        node_type: str,
        utilization: float,
        latency: float,
        distance: int,
        load_factor: float,
    ) -> float:
        """Рассчитывает оценку маршрута через узел."""
        params = self.node_types[node_type]
        latency_factor = latency / params["base_latency"]
        utilization_factor = utilization if utilization < 1 else 2
        distance_factor = distance / 1000
        noise = np.random.normal(0, 0.2)

        weight_factors = {
            NodeType.fiber: {"latency": 0.3, "utilization": 0.3, "distance": 0.2, "load": 0.1, "noise": 0.1},
            NodeType.gen5: {"latency": 0.25, "utilization": 0.35, "distance": 0.2, "load": 0.1, "noise": 0.1},
            NodeType.satellite: {"latency": 0.1, "utilization": 0.3, "distance": 0.4, "load": 0.1, "noise": 0.1},
            NodeType.microwave: {"latency": 0.2, "utilization": 0.3, "distance": 0.3, "load": 0.1, "noise": 0.1},
            NodeType.starlink: {"latency": 0.15, "utilization": 0.3, "distance": 0.35, "load": 0.1, "noise": 0.1},
            NodeType.hybrid: {"latency": 0.25, "utilization": 0.3, "distance": 0.25, "load": 0.1, "noise": 0.1},
        }
        weights = weight_factors.get(
            node_type,
            {"latency": 0.25, "utilization": 0.25, "distance": 0.25, "load": 0.15, "noise": 0.1},
        )

        return (
            latency_factor * weights["latency"]
            + utilization_factor * weights["utilization"]
            + distance_factor * weights["distance"]
            + load_factor * weights["load"]
            + noise * weights["noise"]
        )

    def _simulate_switch(
        self,
        node_id: str,
        node_type: str,
        utilization: float,
        latency: float,
        current_distance: int,
        load_factor: float,
    ) -> dict | None:
        """Симулирует переключение узла на основе оптимальности маршрута и сетевых событий."""
        thresholds = {
            NodeType.fiber: {"utilization": 0.6, "latency": 30},
            NodeType.gen5: {"utilization": 0.65, "latency": 40},
            NodeType.satellite: {"utilization": 0.7, "latency": 650},
            NodeType.microwave: {"utilization": 0.65, "latency": 50},
            NodeType.starlink: {"utilization": 0.65, "latency": 150},
            NodeType.hybrid: {"utilization": 0.65, "latency": 80},
        }
        thresholds_for_type = thresholds.get(node_type, {"utilization": 0.65, "latency": 100})
        params = self.node_types[node_type]

        current_score = self._calculate_route_score(
            node_id,
            node_type,
            utilization,
            latency,
            current_distance,
            load_factor,
        )

        random_trigger = np.random.random() < 0.4

        if (
            utilization > thresholds_for_type["utilization"]
            or latency > thresholds_for_type["latency"]
            or not self.node_status[node_id]["available"]
            or random_trigger
        ):
            alternative_nodes = [
                (n_id, n_type)
                for n_id, n_type in self.node_type_map.items()
                if n_id != node_id and self.node_status[n_id]["available"]
            ]

            if alternative_nodes:
                best_alternative = None
                best_score = float("inf")

                for alt_id, alt_type in alternative_nodes:
                    alt_params = self.node_types[alt_type]
                    alt_distance = self.node_distances[alt_id]
                    alt_load = self.node_status[alt_id]["load_factor"]

                    alt_score = self._calculate_route_score(
                        alt_id,
                        alt_type,
                        0.5,
                        alt_params["base_latency"],
                        alt_distance,
                        alt_load,
                    )

                    if alt_score < best_score:
                        best_score = alt_score
                        best_alternative = (alt_id, alt_type)

                if best_alternative and best_score < current_score:
                    alt_id, alt_type = best_alternative
                    alt_params = self.node_types[alt_type]

                    switch_time = params["switch_time"] * np.random.uniform(0.9, 1.1)
                    switch_packet_loss = params["switch_loss"] * np.random.uniform(0.9, 1.1)

                    if not self.node_status[node_id]["available"]:
                        reason = "Сбой узла"
                    elif utilization > thresholds_for_type["utilization"]:
                        reason = "Высокая загрузка"
                    elif latency > thresholds_for_type["latency"]:
                        reason = "Высокая задержка"
                    else:
                        # Более разнообразные причины переключения
                        reasons = [
                            "Оптимизация маршрута",
                            "Плановое переключение",
                            "Балансировка нагрузки",
                            "Техническое обслуживание",
                            "Оптимизация задержки",
                        ]

                        # Специфичные причины для разных типов узлов
                        if node_type == NodeType.satellite:
                            reasons.extend(["Помехи сигнала", "Ухудшение погодных условий", "Коррекция орбиты"])
                        elif node_type == NodeType.gen5:
                            reasons.extend(["Интерференция сигнала", "Переключение сектора", "Оптимизация MIMO"])
                        elif node_type == NodeType.microwave:
                            reasons.extend(["Атмосферные помехи", "Флуктуации сигнала"])
                        elif node_type == NodeType.starlink:
                            reasons.extend(["Переключение спутника", "Оптимизация геометрии"])

                        reason = np.random.choice(reasons)

                    switch_info = {
                        "timestamp": datetime.now(UTC),
                        "from": node_id,
                        "to": alt_id,
                        "reason": reason,
                        "switch_time": switch_time,
                        "switch_packet_loss": switch_packet_loss,
                    }

                    if node_id not in self.switch_history:
                        self.switch_history[node_id] = []
                    self.switch_history[node_id].append(switch_info)

                    return {
                        "target_node": alt_id,
                        "target_type": alt_type,
                        "reason": reason,
                        "switch_time": switch_time,
                        "switch_packet_loss": switch_packet_loss,
                    }

        return None

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
            created_days_ago = (datetime.now(UTC) - opt["created_at"]).days

            efficacy_multiplier = max(0.5, 1.0 - (created_days_ago * 0.1))

            if "балансировка" in action_type or "load balancing" in action_type:
                reduction_factor = np.random.uniform(0.7, 0.9) * efficacy_multiplier
                if metrics["bandwidth"] / metrics["capacity_mbps"] > 0.7:
                    updated_metrics["bandwidth"] *= reduction_factor

            elif "qos" in action_type or "приоритизация" in action_type:
                latency_improvement = np.random.uniform(0.8, 0.95) * efficacy_multiplier
                packet_loss_improvement = np.random.uniform(0.6, 0.85) * efficacy_multiplier
                updated_metrics["latency"] *= latency_improvement
                updated_metrics["packet_loss"] *= packet_loss_improvement

            elif "переключение" in action_type or "switch" in action_type:
                switch_improvement = 0.7 * efficacy_multiplier
                updated_metrics["switch_time"] *= switch_improvement
                updated_metrics["switch_packet_loss"] *= switch_improvement

            elif "пропускная способность" in action_type or "capacity" in action_type:
                capacity_improvement = np.random.uniform(1.05, 1.15) * efficacy_multiplier
                updated_metrics["capacity_mbps"] *= capacity_improvement

            elif "маршрутизация" in action_type or "routing" in action_type:
                if node_type == NodeType.hybrid:
                    updated_metrics["latency"] *= 0.85 * efficacy_multiplier

            elif "шейпинг" in action_type or "shaping" in action_type:
                if updated_metrics["jitter"] > 5:
                    updated_metrics["jitter"] *= 0.8 * efficacy_multiplier

            elif "отказоустойчивость" in action_type or "failover" in action_type:
                updated_metrics["packet_loss"] *= 0.85 * efficacy_multiplier

            self.node_status[node_id]["failure_reduction"] = 0.5

        return updated_metrics

    async def collect_data(self, nodes: list[str], timestamp: datetime) -> list[dict]:
        """Собирает данные из датасета с симуляцией нагрузки и переключений."""
        all_nodes = await self._get_nodes_from_db()

        self.node_type_map = {node["node_id"]: node["node_type"] for node in all_nodes}

        if not self.node_to_router:
            self._initialize_node_mappings(all_nodes)

        target_nodes = [node for node in all_nodes if node["node_id"] in nodes]
        if not target_nodes:
            target_nodes = all_nodes

        self.index = (self.index + 1) % len(self.data)

        result = []

        for node in target_nodes:
            node_id = node["node_id"]
            node_type = node["node_type"]
            params = self.node_types[node_type]

            status = self._simulate_network_events(node_id, node_type)

            bandwidth = 0.0
            latency = 0.0
            packet_loss = 0.0
            jitter = 0.0
            switched_from = None
            switch_reason = None
            switch_time = 0.0
            switch_packet_loss = 0.0
            signal_strength = None
            interference_level = None
            error_rate = None

            if not status["available"]:
                switch_info = self._simulate_switch(
                    node_id,
                    node_type,
                    utilization=1.0,
                    latency=params["base_latency"],
                    current_distance=self.node_distances[node_id],
                    load_factor=status["load_factor"],
                )

                if switch_info:
                    target_id = switch_info["target_node"]
                    target_type = switch_info["target_type"]
                    target_params = self.node_types[target_type]

                    router_data = self.data[self.data["router_id"] == self.node_to_router[target_id]]
                    if not router_data.empty:
                        row = router_data.iloc[self.index % len(router_data)]
                        base_traffic = float(row["traffic_mbps"])
                    else:
                        base_traffic = target_params["base_bandwidth"] * 0.5

                    scale_factor = {
                        NodeType.fiber: 100,
                        NodeType.gen5: 50,
                        NodeType.satellite: 10,
                        NodeType.microwave: 30,
                        NodeType.starlink: 20,
                        NodeType.hybrid: 80,
                    }.get(target_type, 50)

                    noise = np.random.normal(0, 0.1 * base_traffic)

                    hour_factor = 0.7 + 0.6 * np.sin(np.pi * (timestamp.hour % 24) / 12)

                    bandwidth = max(0, base_traffic * scale_factor * hour_factor * status["load_factor"] + noise)
                    bandwidth = min(bandwidth, target_params["capacity"] * 1.2)

                    utilization = bandwidth / target_params["capacity"]
                    latency = target_params["base_latency"] * (1 + 2 * utilization)
                    packet_loss = target_params["base_packet_loss"] * (1 + 3 * utilization)
                    jitter = target_params["jitter"] * (1 + utilization * 1.2 + np.random.normal(0, 0.1))

                    switched_from = node_id
                    switch_reason = switch_info["reason"]
                    switch_time = switch_info["switch_time"]
                    switch_packet_loss = switch_info["switch_packet_loss"]

                    if target_type in [NodeType.satellite, NodeType.gen5, NodeType.microwave, NodeType.starlink]:
                        signal_strength = -60 - np.random.uniform(0, 40)
                        interference_level = np.random.uniform(0, 15)
                        error_rate = packet_loss / 100 * np.random.uniform(0.8, 1.5)

                    packet_loss += switch_packet_loss
                else:
                    switch_reason = "Сбой сети: нет доступных узлов"
            else:
                router_data = self.data[self.data["router_id"] == self.node_to_router[node_id]]
                if not router_data.empty:
                    row = router_data.iloc[self.index % len(router_data)]
                    base_traffic = float(row["traffic_mbps"])
                else:
                    base_traffic = params["base_bandwidth"] * 0.5

                scale_factor = {
                    NodeType.fiber: 100,
                    NodeType.gen5: 50,
                    NodeType.satellite: 10,
                    NodeType.microwave: 30,
                    NodeType.starlink: 20,
                    NodeType.hybrid: 80,
                }.get(node_type, 50)

                noise = np.random.normal(0, 0.1 * base_traffic)

                hour_factor = 0.7 + 0.6 * np.sin(np.pi * (timestamp.hour % 24) / 12)

                bandwidth = max(0, base_traffic * scale_factor * hour_factor * status["load_factor"] + noise)
                bandwidth = min(bandwidth, params["capacity"] * 1.2)

                utilization = bandwidth / params["capacity"]
                latency = params["base_latency"] * (1 + 2 * utilization)
                packet_loss = params["base_packet_loss"] * (1 + 3 * utilization)
                jitter = params["jitter"] * (1 + utilization * 1.2 + np.random.normal(0, 0.1))

                if node_type in [NodeType.satellite, NodeType.gen5, NodeType.microwave, NodeType.starlink]:
                    weather_factor = np.random.uniform(0.8, 1.2)
                    latency *= weather_factor
                    packet_loss *= weather_factor

                    signal_strength = -60 - np.random.uniform(0, 40)
                    interference_level = np.random.uniform(0, 15)
                    error_rate = packet_loss / 100 * np.random.uniform(0.8, 1.5)

                distance = self.node_distances[node_id]
                switch_info = self._simulate_switch(
                    node_id,
                    node_type,
                    utilization,
                    latency,
                    distance,
                    status["load_factor"],
                )

                if switch_info:
                    switched_from = node_id
                    switch_reason = switch_info["reason"]
                    switch_time = switch_info["switch_time"]
                    switch_packet_loss = switch_info["switch_packet_loss"]
                    packet_loss += switch_packet_loss

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

        return result
