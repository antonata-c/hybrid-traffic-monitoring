import logging
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import settings
from enums import NodeType
from .base import DataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDataSource(DataSource):
    """Источник данных, использующий датасет Abilene с симуляцией переключений и разнообразной нагрузкой."""

    def __init__(self):
        self.node_types = {
            NodeType.fiber: {
                "base_bandwidth": 1000,
                "base_latency": 5,
                "base_packet_loss": 0.1,
                "capacity": 1000,
                "switch_time": 0.1,
                "switch_loss": 0.5
            },
            NodeType.satellite: {
                "base_bandwidth": 50,
                "base_latency": 600,
                "base_packet_loss": 2,
                "capacity": 100,
                "switch_time": 0.5,
                "switch_loss": 2.0
            },
            NodeType.gen5: {
                "base_bandwidth": 500,
                "base_latency": 20,
                "base_packet_loss": 1,
                "capacity": 500,
                "switch_time": 0.2,
                "switch_loss": 1.0
            }
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
                    "traffic_mbps": [10.0 * (i % 100 + 1) for i in range(1000)]
                }
            )
        self.node_to_router = {node: f"router_{i % 12}" for i, node in enumerate(settings.NETWORK_NODES)}
        self.index = 0
        self.switch_history: Dict[str, List[Dict]] = {}
        self.node_distances = {node: np.random.randint(10, 1000) for node in settings.NETWORK_NODES}
        self.node_status: Dict[str, Dict] = {node: {"available": True, "load_factor": 1.0} for node in
                                             settings.NETWORK_NODES}

    def _get_node_type(self, node: str) -> NodeType:
        """Возвращает тип узла."""
        if "fiber" in node.lower():
            return NodeType.fiber
        elif "sat" in node.lower():
            return NodeType.satellite
        return NodeType.gen5

    def _simulate_network_events(self, node: str, node_type: NodeType) -> Dict:
        """Симулирует сетевые события: сбои, пики нагрузки, ухудшение метрик."""
        status = self.node_status[node]
        failure_prob = 0.001 if node_type == NodeType.satellite else 0.03
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
        node: str,
        node_type: NodeType,
        utilization: float,
        latency: float,
        distance: int,
        load_factor: float
    ) -> float:
        """Рассчитывает оценку маршрута через узел."""
        params = self.node_types[node_type]
        latency_factor = latency / params["base_latency"]
        utilization_factor = utilization if utilization < 1 else 2
        distance_factor = distance / 1000
        noise = np.random.normal(0, 0.2)
        load_factor = load_factor if node_type != NodeType.satellite else load_factor * 0.5
        return (
            latency_factor * 0.25 + utilization_factor * 0.25 + distance_factor * 0.25 + load_factor * 0.15 + noise * 0.1)

    def _simulate_switch(
        self,
        node: str,
        node_type: NodeType,
        utilization: float,
        latency: float,
        current_distance: int,
        load_factor: float
    ) -> Optional[Dict]:
        """Симулирует переключение узла на основе оптимальности маршрута и сетевых событий."""
        thresholds = {
            NodeType.fiber: {"utilization": 0.6, "latency": 30},
            NodeType.satellite: {"utilization": 0.7, "latency": 650},
            NodeType.gen5: {"utilization": 0.65, "latency": 40}
        }
        params = self.node_types[node_type]

        current_score = self._calculate_route_score(
            node,
            node_type,
            utilization,
            latency,
            current_distance,
            load_factor
        )
        random_trigger = np.random.random() < 0.4
        if (random_trigger or utilization > thresholds[node_type]["utilization"] or
            latency > thresholds[node_type]["latency"]):
            route_scores = []
            prev_from_node = self.switch_history.get(node, [{}])[-1].get("from_node")
            for other_node in settings.NETWORK_NODES:
                if other_node == node or other_node == prev_from_node:
                    continue
                other_type = self._get_node_type(other_node)
                other_status = self._simulate_network_events(other_node, other_type)
                if not other_status["available"]:
                    continue
                other_params = self.node_types[other_type]
                router_data = self.data[self.data["router_id"] == self.node_to_router.get(other_node, "router_0")]
                base_traffic = float(
                    router_data.iloc[self.index % len(router_data)]["traffic_mbps"]
                ) if not router_data.empty else other_params["base_bandwidth"]
                scale_factor = {NodeType.fiber: 100, NodeType.gen5: 50, NodeType.satellite: 10}[other_type]
                other_bandwidth = min(
                    base_traffic * scale_factor * other_status["load_factor"],
                    other_params["capacity"]
                )
                other_utilization = other_bandwidth / other_params["capacity"]
                other_latency = other_params["base_latency"] * (1 + 2 * other_utilization)
                distance = self.node_distances[other_node]
                score = self._calculate_route_score(
                    other_node,
                    other_type,
                    other_utilization,
                    other_latency,
                    distance,
                    other_status["load_factor"]
                )
                route_scores.append((other_node, other_type, score))

            if not route_scores:
                return None

            scores = np.array([score for _, _, score in route_scores])
            probabilities = np.exp(-scores) / np.sum(np.exp(-scores))
            target_idx = np.random.choice(len(route_scores), p=probabilities)
            target_node, target_type, target_score = route_scores[target_idx]

            if target_score >= current_score and not random_trigger:
                return None

            reason = "Перегрузка" if utilization > thresholds[node_type][
                "utilization"] else "Высокая задержка" if latency > thresholds[node_type][
                "latency"] else "Случайное переключение"
            switch_event = {
                "from_node": node,
                "to_node": target_node,
                "reason": reason,
                "time": params["switch_time"],
                "loss": params["switch_loss"],
                "load_factor": load_factor,
                "event": self.node_status[node].get("event")
            }
            self.switch_history.setdefault(node, []).append(switch_event)
            return switch_event
        return None

    async def collect_data(self, nodes: List[str], timestamp: datetime) -> List[Dict]:
        """Собирает данные из датасета с симуляцией нагрузки и переключений."""
        data = []
        self.index = (self.index + 1) % len(self.data)

        for node in nodes:
            node_type = self._get_node_type(node)
            params = self.node_types[node_type]
            status = self._simulate_network_events(node, node_type)

            bandwidth_mbps = 0.0
            latency = 0.0
            packet_loss = 0.0
            switched_from = None
            switch_reason = None
            switch_time = 0.0
            switch_packet_loss = 0.0

            if not status["available"]:

                switch_info = self._simulate_switch(
                    node, node_type, utilization=1.0, latency=params["base_latency"],
                    current_distance=self.node_distances[node], load_factor=status["load_factor"]
                )
                if switch_info:
                    switched_from = switch_info["to_node"]
                    switch_reason = "Сбой сети"
                    switch_time = switch_info["time"]
                    switch_packet_loss = switch_info["loss"]

                    target_node = switched_from
                    target_type = self._get_node_type(target_node)
                    target_params = self.node_types[target_type]
                    target_status = self._simulate_network_events(target_node, target_type)

                    if target_status["available"]:
                        router_data = self.data[
                            self.data["router_id"] == self.node_to_router.get(target_node, "router_0")]
                        base_traffic = float(
                            router_data.iloc[self.index % len(router_data)]["traffic_mbps"]
                        ) if not router_data.empty else target_params["base_bandwidth"]
                        scale_factor = {
                            NodeType.fiber: 100,
                            NodeType.gen5: 50,
                            NodeType.satellite: 10
                        }[target_type]
                        noise = np.random.normal(0, 0.1 * base_traffic)
                        time_factor = 1 + 0.2 * np.sin(self.index / 10)
                        bandwidth_mbps = max(
                            0,
                            base_traffic * scale_factor * time_factor * target_status["load_factor"] + noise
                        )
                        bandwidth_mbps = min(bandwidth_mbps, target_params["capacity"] * 1.5)
                        utilization = bandwidth_mbps / target_params["capacity"]
                        latency = target_params["base_latency"] * (1 + 2 * utilization)
                        packet_loss = target_params["base_packet_loss"] * (1 + 3 * utilization)

                        bandwidth_mbps *= 0.9
                        latency *= 1.2
                        packet_loss += switch_packet_loss
                    else:

                        switch_reason = "Сбой сети: нет доступных узлов"
                else:
                    switch_reason = "Сбой сети: нет доступных узлов"
            else:

                router_data = self.data[self.data["router_id"] == self.node_to_router[node]]
                if not router_data.empty:
                    row = router_data.iloc[self.index % len(router_data)]
                    base_traffic = float(row["traffic_mbps"])
                else:
                    base_traffic = params["base_bandwidth"] * 0.5

                scale_factor = {
                    NodeType.fiber: 100,
                    NodeType.gen5: 50,
                    NodeType.satellite: 10
                }[node_type]
                noise = np.random.normal(0, 0.1 * base_traffic)
                time_factor = 1 + 0.2 * np.sin(self.index / 10)
                bandwidth_mbps = max(0, base_traffic * scale_factor * time_factor * status["load_factor"] + noise)
                bandwidth_mbps = min(bandwidth_mbps, params["capacity"] * 1.5)

                utilization = bandwidth_mbps / params["capacity"]
                latency = params["base_latency"] * (1 + 2 * utilization)
                packet_loss = params["base_packet_loss"] * (1 + 3 * utilization)

                distance = self.node_distances[node]
                switch_info = self._simulate_switch(
                    node,
                    node_type,
                    utilization,
                    latency,
                    distance,
                    status["load_factor"]
                )
                if switch_info:
                    switched_from = switch_info["to_node"]
                    switch_reason = switch_info["reason"]
                    switch_time = switch_info["time"]
                    switch_packet_loss = switch_info["loss"]
                    bandwidth_mbps *= 0.9
                    latency *= 1.2
                    packet_loss += switch_packet_loss

            data.append(
                {
                    "node_id": node,
                    "node_type": node_type.value,
                    "bandwidth": round(bandwidth_mbps, 2),
                    "capacity_mbps": params["capacity"],
                    "latency": round(latency, 2),
                    "packet_loss": round(packet_loss, 2),
                    "switched_from": switched_from,
                    "switch_reason": switch_reason,
                    "switch_time": round(switch_time, 2),
                    "switch_packet_loss": round(switch_packet_loss, 2),
                    "timestamp": timestamp
                }
            )

        return data
