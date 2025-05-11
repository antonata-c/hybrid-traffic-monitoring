from datetime import datetime

import numpy as np

from config import settings
from .base import DataSource


class CalculatedDataSource(DataSource):
    """Источник данных, генерирующий метрики на основе расчетов."""

    def __init__(self):
        self.node_types = {
            "fiber": {"base_bandwidth": 1000, "base_latency": 5, "base_packet_loss": 0.1, "capacity": 1000},
            "satellite": {"base_bandwidth": 50, "base_latency": 600, "base_packet_loss": 2, "capacity": 100},
            "5G": {"base_bandwidth": 500, "base_latency": 20, "base_packet_loss": 1, "capacity": 500}
        }
        self.total_traffic_demand = settings.TOTAL_TRAFFIC_DEMAND
        self.traffic_shares = {node: 1.0 / len(settings.NETWORK_NODES) for node in settings.NETWORK_NODES}

    def _get_node_type(self, node: str) -> str:
        """Возвращает тип узла."""
        if "fiber" in node.lower():
            return "fiber"
        elif "sat" in node.lower():
            return "satellite"
        return "5G"

    async def collect_data(self, nodes: list[str], timestamp: datetime) -> list[dict]:
        """Собирает симулированные данные о трафике."""
        data = []
        is_peak_hour = 18 <= timestamp.hour <= 22

        for node in nodes:
            node_type = self._get_node_type(node)
            params = self.node_types[node_type]
            assigned_bandwidth = self.total_traffic_demand * self.traffic_shares[node]
            utilization = min(assigned_bandwidth / params["capacity"], 1.0)
            latency = params["base_latency"] * (1 + 2 * utilization)
            packet_loss = params["base_packet_loss"] * (1 + 3 * utilization)
            bandwidth_mbps = assigned_bandwidth * (1.2 if is_peak_hour else 1.0)

            switch_probability = 0.1
            switched_from = None
            switch_reason = None
            switch_time = 0.0
            switch_packet_loss = 0.0
            if np.random.random() < switch_probability:
                switch_time = np.random.uniform(0.1, 0.5)
                switch_packet_loss = np.random.uniform(0.5, 2.0)
                switch_reason = "optimization"
                other_nodes = [n for n in nodes if n != node]
                if other_nodes:
                    switched_from = np.random.choice(other_nodes)

            data.append(
                {
                    "node_id": node,
                    "node_type": node_type,
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
