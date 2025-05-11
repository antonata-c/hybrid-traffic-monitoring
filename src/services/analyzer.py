import logging
from datetime import UTC, datetime, timedelta
from typing import Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from enums import NodeType
from models import Traffic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficAnalyzer:
    """Класс для глубокого анализа данных о трафике в гибридных сетях, без оптимизаций."""

    def __init__(self, db: AsyncSession):
        self.db = db

        self.latency_thresholds = {"fiber": 50, "satellite": 700, "5G": 50}
        self.packet_loss_thresholds = {"fiber": 2, "satellite": 5, "5G": 2}
        self.switch_time_threshold = 0.3
        self.utilization_threshold_high = 0.8
        self.utilization_threshold_low = 0.5

    async def _fetch_traffic_data(self, minutes: int = 60, node_type: NodeType = None) -> Sequence[Traffic]:
        """Получает данные о трафике за последние N минут.

        Args:
            minutes: Период анализа в минутах.
            node_type: Тип узла для фильтрации (fiber, satellite, 5G), опционально.

        Returns:
            Список записей трафика.
        """
        query = select(Traffic).filter(Traffic.timestamp >= datetime.now(UTC) - timedelta(minutes=minutes))
        if node_type:
            query = query.filter(Traffic.node_type == node_type)
        result = await self.db.scalars(query)
        return result.all()

    @staticmethod
    async def _calculate_trends(traffic_data: Sequence[Traffic], metric: str) -> dict[str, float]:
        """Рассчитывает тренды для указанной метрики по узлам с использованием линейной регрессии.

        Args:
            traffic_data: Список записей трафика.
            metric: Метрика для анализа (bandwidth, latency, packet_loss, switch_time).

        Returns:
            Словарь {node_id: slope}, где slope — скорость изменения метрики (положительная = рост).
        """
        trends = {}
        for node in set(t.node_id for t in traffic_data):
            node_data = sorted(
                [t for t in traffic_data if t.node_id == node],
                key=lambda x: x.timestamp
            )
            if len(node_data) > 5:
                values = [getattr(t, metric) for t in node_data]
                times = [(t.timestamp - node_data[0].timestamp).total_seconds() / 60 for t in node_data]
                try:
                    slope, _ = np.polyfit(times, values, 1)
                    trends[node] = round(float(slope), 4)
                except np.linalg.LinAlgError:
                    trends[node] = 0.0
        return trends

    async def _detect_anomalies(self, traffic_data: Sequence[Traffic], metric: str) -> list[dict]:
        """Выявляет аномалии для указанной метрики на основе пороговых значений.

        Args:
            traffic_data: Список записей трафика.
            metric: Метрика для анализа (latency, packet_loss, switch_time).

        Returns:
            Список аномалий с информацией о узле, значении и времени.
        """
        anomalies = []
        for t in traffic_data:
            value = getattr(t, metric)
            if metric == "latency" and value > self.latency_thresholds[t.node_type] * 1.5:
                anomalies.append(
                    {
                        "node_id": t.node_id,
                        "metric": metric,
                        "value": round(value, 2),
                        "threshold": self.latency_thresholds[t.node_type],
                        "timestamp": t.timestamp.isoformat()
                    }
                )
            elif metric == "packet_loss" and value > self.packet_loss_thresholds[t.node_type] * 1.5:
                anomalies.append(
                    {
                        "node_id": t.node_id,
                        "metric": metric,
                        "value": round(value, 2),
                        "threshold": self.packet_loss_thresholds[t.node_type],
                        "timestamp": t.timestamp.isoformat()
                    }
                )
            elif metric == "switch_time" and value > self.switch_time_threshold * 1.5:
                anomalies.append(
                    {
                        "node_id": t.node_id,
                        "metric": metric,
                        "value": round(value, 2),
                        "threshold": self.switch_time_threshold,
                        "timestamp": t.timestamp.isoformat()
                    }
                )
        return anomalies

    @staticmethod
    async def _forecast_metric(traffic_data: Sequence[Traffic], metric: str, steps: int = 5) -> dict[str, float]:
        """Прогнозирует значение метрики с использованием скользящего среднего.

        Args:
            traffic_data: Список записей трафика.
            metric: Метрика для прогнозирования (bandwidth, latency).
            steps: Количество точек для скользящего среднего.

        Returns:
            Словарь {node_id: forecasted_value} с прогнозами.
        """
        forecasts = {}
        for node in set(t.node_id for t in traffic_data):
            node_data = sorted(
                [t for t in traffic_data if t.node_id == node],
                key=lambda x: x.timestamp
            )
            if len(node_data) >= steps:
                values = [getattr(t, metric) for t in node_data[-steps:]]
                forecast = sum(values) / len(values)
                forecasts[node] = round(forecast, 2)
        return forecasts

    @staticmethod
    async def _calculate_correlations(traffic_data: Sequence[Traffic]) -> dict[str, float]:
        """Рассчитывает корреляции между ключевыми метриками.

        Args:
            traffic_data: Список записей трафика.

        Returns:
            Словарь с коэффициентами корреляции.
        """
        if len(traffic_data) < 2:
            return {}
        bandwidth = [t.bandwidth for t in traffic_data]
        latency = [t.latency for t in traffic_data]
        packet_loss = [t.packet_loss for t in traffic_data]
        utilization = [t.bandwidth / t.capacity_mbps for t in traffic_data]
        switch_time = [t.switch_time for t in traffic_data]

        correlations = {}
        try:
            correlations["bandwidth_vs_latency"] = vs_latency if not np.isnan(
                vs_latency := round(
                    float(np.corrcoef(bandwidth, latency, dtype=float)[0, 1]),
                    2
                )
            ) else 0.0
            correlations["utilization_vs_packet_loss"] = vs_packet_loss if not np.isnan(
                vs_packet_loss := round(
                    float(np.corrcoef(utilization, packet_loss, dtype=float)[0, 1]),
                    2
                )
            ) else 0.0
            correlations["switch_time_vs_packet_loss"] = switch_vs_packet_loss if not np.isnan(
                switch_vs_packet_loss := round(
                    float(np.corrcoef(switch_time, packet_loss, dtype=float)[0, 1]),
                    2
                )
            ) else 0.0
        except np.linalg.LinAlgError:
            correlations = {k: 0.0 for k in
                            ["bandwidth_vs_latency", "utilization_vs_packet_loss", "switch_time_vs_packet_loss"]}
        return correlations

    async def get_traffic_analytics(self, node_type: NodeType | None = None, minutes: int = 60) -> dict:
        """Возвращает глубокую аналитику по трафику за указанный период.

        Args:
            node_type: Тип узла для фильтрации (fiber, satellite, 5G), опционально.
            minutes: Период анализа в минутах (по умолчанию 60).

        Returns:
            Словарь с метриками, трендами, аномалиями, прогнозами и корреляциями.
        """
        traffic_data = await self._fetch_traffic_data(minutes=minutes, node_type=node_type)
        if not traffic_data:
            return {"message": "Нет доступных данных"}

        avg_bandwidth = sum(t.bandwidth for t in traffic_data) / len(traffic_data)
        avg_latency = sum(t.latency for t in traffic_data) / len(traffic_data)
        avg_packet_loss = sum(t.packet_loss for t in traffic_data) / len(traffic_data)
        avg_switch_time = sum(t.switch_time for t in traffic_data) / len(traffic_data)
        avg_switch_packet_loss = sum(t.switch_packet_loss for t in traffic_data) / len(traffic_data)

        nodes_by_type = {"fiber": [], "satellite": [], "5G": []}
        for t in traffic_data:
            nodes_by_type[t.node_type].append(t)

        type_metrics = {}
        for node_type, nodes in nodes_by_type.items():
            if nodes:
                type_metrics[node_type] = {
                    "avg_bandwidth": round(sum(t.bandwidth for t in nodes) / len(nodes), 2),
                    "avg_latency": round(sum(t.latency for t in nodes) / len(nodes), 2),
                    "avg_packet_loss": round(sum(t.packet_loss for t in nodes) / len(nodes), 2),
                    "switch_frequency": round(sum(1 for t in nodes if t.switch_time > 0) / len(nodes), 2),
                }

        bandwidth_trends = await self._calculate_trends(traffic_data, "bandwidth")
        latency_trends = await self._calculate_trends(traffic_data, "latency")
        packet_loss_trends = await self._calculate_trends(traffic_data, "packet_loss")
        switch_time_trends = await self._calculate_trends(traffic_data, "switch_time")

        latency_anomalies = await self._detect_anomalies(traffic_data, "latency")
        packet_loss_anomalies = await self._detect_anomalies(traffic_data, "packet_loss")
        switch_time_anomalies = await self._detect_anomalies(traffic_data, "switch_time")

        bandwidth_forecasts = await self._forecast_metric(traffic_data, "bandwidth")
        latency_forecasts = await self._forecast_metric(traffic_data, "latency")

        correlations = await self._calculate_correlations(traffic_data)

        switch_stats = {
            "total_switches": sum(1 for t in traffic_data if t.switch_time > 0),
            "nodes_with_switches": len(set(t.node_id for t in traffic_data if t.switch_time > 0)),
            "avg_switch_time": round(avg_switch_time, 2),
            "avg_switch_packet_loss": round(avg_switch_packet_loss, 2),
        }

        return {
            "message": f"Аналитика трафика за последние {minutes} минут",
            "average_metrics": {
                "bandwidth_mbps": round(avg_bandwidth, 2),
                "latency_ms": round(avg_latency, 2),
                "packet_loss_percent": round(avg_packet_loss, 2),
                "switch_time_seconds": round(avg_switch_time, 2),
                "switch_packet_loss_percent": round(avg_switch_packet_loss, 2),
            },
            "type_metrics": type_metrics,
            "trends": {
                "bandwidth_mbps_per_minute": bandwidth_trends,
                "latency_ms_per_minute": latency_trends,
                "packet_loss_percent_per_minute": packet_loss_trends,
                "switch_time_seconds_per_minute": switch_time_trends,
            },
            "anomalies": {
                "latency": latency_anomalies,
                "packet_loss": packet_loss_anomalies,
                "switch_time": switch_time_anomalies,
            },
            "forecasts": {
                "bandwidth_mbps_next_5min": bandwidth_forecasts,
                "latency_ms_next_5min": latency_forecasts,
            },
            "correlations": correlations,
            "switch_statistics": switch_stats,
        }

    async def get_detailed_node_analytics(self, node_id: str, minutes: int = 60) -> dict:
        """Возвращает детальную аналитику для конкретного узла.

        Args:
            node_id: Идентификатор узла.
            minutes: Период анализа в минутах (по умолчанию 60).

        Returns:
            Словарь с подробной статистикой по узлу.
        """
        query = select(Traffic).filter(
            Traffic.node_id == node_id,
            Traffic.timestamp >= datetime.now(UTC) - timedelta(minutes=minutes)
        )
        result = await self.db.scalars(query)
        node_data = result.all()

        if not node_data:
            return {"message": f"Нет данных для узла {node_id}"}

        bandwidth_values = [t.bandwidth for t in node_data]
        latency_values = [t.latency for t in node_data]
        packet_loss_values = [t.packet_loss for t in node_data]
        utilization_values = [t.bandwidth / t.capacity_mbps for t in node_data]
        switch_time_values = [t.switch_time for t in node_data]

        bandwidth_trend = (await self._calculate_trends(node_data, "bandwidth")).get(node_id, 0.0)
        latency_trend = (await self._calculate_trends(node_data, "latency")).get(node_id, 0.0)

        anomalies = []
        anomalies.extend(await self._detect_anomalies(node_data, "latency"))
        anomalies.extend(await self._detect_anomalies(node_data, "packet_loss"))
        anomalies.extend(await self._detect_anomalies(node_data, "switch_time"))

        return {
            "node_id": node_id,
            "node_type": node_data[0].node_type,
            "metrics": {
                "avg_bandwidth_mbps": round(sum(bandwidth_values) / len(bandwidth_values), 2),
                "avg_latency_ms": round(sum(latency_values) / len(latency_values), 2),
                "avg_packet_loss_percent": round(sum(packet_loss_values) / len(packet_loss_values), 2),
                "avg_utilization": round(sum(utilization_values) / len(node_data), 2),
                "avg_switch_time_seconds": round(sum(switch_time_values) / len(node_data), 2),
            },
            "peak_values": {
                "max_bandwidth_mbps": round(max(bandwidth_values), 2),
                "max_latency_ms": round(max(latency_values), 2),
                "max_packet_loss_percent": round(max(packet_loss_values), 2),
                "max_utilization": round(max(utilization_values), 2),
            },
            "trends": {
                "bandwidth_mbps_per_minute": bandwidth_trend,
                "latency_ms_per_minute": latency_trend,
            },
            "anomalies": anomalies,
            "switch_statistics": {
                "total_switches": sum(1 for t in node_data if t.switch_time > 0),
                "avg_switch_time_seconds": round(sum(t.switch_time for t in node_data) / len(node_data), 2),
                "avg_switch_packet_loss_percent": round(
                    sum(t.switch_packet_loss for t in node_data) / len(node_data),
                    2
                ),
            },
        }
