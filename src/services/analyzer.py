import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from enums import NetworkStatus, NodeType, TrafficType
from models import NetworkLink, Traffic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficAnalyzer:
    """Расширенный класс для анализа данных о трафике в гибридных сетях."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.latency_thresholds = {
            "fiber": settings.LATENCY_THRESHOLD_FIBER,
            "satellite": settings.LATENCY_THRESHOLD_SATELLITE,
            "5G": settings.LATENCY_THRESHOLD_5G,
            "microwave": 50,
            "starlink": 200,
            "hybrid": 100,
        }
        self.packet_loss_thresholds = {
            "fiber": 1,
            "satellite": 3,
            "5G": 2,
            "microwave": 2,
            "starlink": 3,
            "hybrid": 2,
        }
        self.switch_time_threshold = 0.3
        self.utilization_threshold_high = settings.HIGH_UTILIZATION_THRESHOLD / 100.0
        self.utilization_threshold_low = settings.LOW_UTILIZATION_THRESHOLD / 100.0
        self.anomaly_sensitivity = settings.ANOMALY_DETECTION_SENSITIVITY
        self.qos_metrics = {
            TrafficType.VOICE: {"max_latency": 100, "max_jitter": 30, "max_packet_loss": 1},
            TrafficType.VIDEO: {"max_latency": 150, "max_jitter": 50, "max_packet_loss": 1.5},
            TrafficType.INTERACTIVE: {"max_latency": 200, "max_jitter": 100, "max_packet_loss": 2},
            TrafficType.STREAMING: {"max_latency": 400, "max_jitter": 150, "max_packet_loss": 3},
            TrafficType.DATA: {"max_latency": 1000, "max_jitter": 300, "max_packet_loss": 5},
            TrafficType.IOT: {"max_latency": 500, "max_jitter": 200, "max_packet_loss": 3},
            TrafficType.SIGNALING: {"max_latency": 100, "max_jitter": 50, "max_packet_loss": 0.5},
        }

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
        for node in {t.node_id for t in traffic_data}:
            node_data = sorted(
                [t for t in traffic_data if t.node_id == node],
                key=lambda x: x.timestamp,
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
            if metric == "latency" and value > self.latency_thresholds[t.node_type] * self.anomaly_sensitivity:
                anomalies.append(
                    {
                        "node_id": t.node_id,
                        "metric": metric,
                        "value": round(value, 2),
                        "threshold": self.latency_thresholds[t.node_type],
                        "timestamp": t.timestamp.isoformat(),
                    },
                )
            elif (
                metric == "packet_loss" and value > self.packet_loss_thresholds[t.node_type] * self.anomaly_sensitivity
            ):
                anomalies.append(
                    {
                        "node_id": t.node_id,
                        "metric": metric,
                        "value": round(value, 2),
                        "threshold": self.packet_loss_thresholds[t.node_type],
                        "timestamp": t.timestamp.isoformat(),
                    },
                )
            elif metric == "switch_time" and value > self.switch_time_threshold * self.anomaly_sensitivity:
                anomalies.append(
                    {
                        "node_id": t.node_id,
                        "metric": metric,
                        "value": round(value, 2),
                        "threshold": self.switch_time_threshold,
                        "timestamp": t.timestamp.isoformat(),
                    },
                )
            elif metric == "jitter" and hasattr(t, "jitter") and t.jitter > 100 * self.anomaly_sensitivity:
                anomalies.append(
                    {
                        "node_id": t.node_id,
                        "metric": metric,
                        "value": round(t.jitter, 2),
                        "threshold": 100,
                        "timestamp": t.timestamp.isoformat(),
                    },
                )
        return anomalies

    async def _forecast_metric(self, traffic_data: Sequence[Traffic], metric: str, steps: int = 5) -> dict[str, float]:
        """Прогнозирует значение метрики с использованием ARIMA или скользящего среднего.

        Args:
            traffic_data: Список записей трафика.
            metric: Метрика для прогнозирования (bandwidth, latency).
            steps: Количество точек для прогноза.

        Returns:
            Словарь {node_id: forecasted_value} с прогнозами.
        """
        forecasts = {}
        for node in {t.node_id for t in traffic_data}:
            node_data = sorted(
                [t for t in traffic_data if t.node_id == node],
                key=lambda x: x.timestamp,
            )
            if len(node_data) >= max(steps, 10):
                values = [getattr(t, metric) for t in node_data]
                try:
                    if len(values) >= 20:
                        x = np.arange(len(values))
                        y = np.array(values)
                        a, b = np.polyfit(x, y, 1)
                        forecast = a * (len(values) + steps) + b
                    else:
                        forecast = sum(values[-steps:]) / steps
                    forecasts[node] = round(float(forecast), 2)
                except Exception as e:
                    logger.warning(f"Ошибка прогнозирования для узла {node}: {e}")
                    forecasts[node] = round(sum(values[-3:]) / 3, 2)
        return forecasts

    @staticmethod
    async def _calculate_correlations(traffic_data: Sequence[Traffic]) -> dict[str, float]:
        """Рассчитывает корреляции между ключевыми метриками в гибридной сети.

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
        jitter = [getattr(t, "jitter", 0) for t in traffic_data]

        correlations = {}
        try:
            correlations["bandwidth_vs_latency"] = (
                vs_latency
                if not np.isnan(
                    vs_latency := round(
                        float(np.corrcoef(bandwidth, latency, dtype=float)[0, 1]),
                        2,
                    ),
                )
                else 0.0
            )
            correlations["utilization_vs_packet_loss"] = (
                vs_packet_loss
                if not np.isnan(
                    vs_packet_loss := round(
                        float(np.corrcoef(utilization, packet_loss, dtype=float)[0, 1]),
                        2,
                    ),
                )
                else 0.0
            )
            correlations["switch_time_vs_packet_loss"] = (
                switch_vs_packet_loss
                if not np.isnan(
                    switch_vs_packet_loss := round(
                        float(np.corrcoef(switch_time, packet_loss, dtype=float)[0, 1]),
                        2,
                    ),
                )
                else 0.0
            )
            if any(j > 0 for j in jitter):
                correlations["jitter_vs_latency"] = (
                    jitter_vs_latency
                    if not np.isnan(
                        jitter_vs_latency := round(
                            float(np.corrcoef(jitter, latency, dtype=float)[0, 1]),
                            2,
                        ),
                    )
                    else 0.0
                )
        except np.linalg.LinAlgError:
            correlations = dict.fromkeys(
                ["bandwidth_vs_latency", "utilization_vs_packet_loss", "switch_time_vs_packet_loss"],
                0.0,
            )
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
        avg_jitter = sum(getattr(t, "jitter", 0) for t in traffic_data) / len(traffic_data)
        avg_utilization = sum(t.bandwidth / t.capacity_mbps for t in traffic_data) / len(traffic_data)
        node_type_data = {}
        for node_type in {t.node_type for t in traffic_data}:
            type_traffic = [t for t in traffic_data if t.node_type == node_type]
            if type_traffic:
                node_type_data[node_type] = {
                    "avg_bandwidth": sum(t.bandwidth for t in type_traffic) / len(type_traffic),
                    "avg_latency": sum(t.latency for t in type_traffic) / len(type_traffic),
                    "avg_packet_loss": sum(t.packet_loss for t in type_traffic) / len(type_traffic),
                    "avg_utilization": sum(t.bandwidth / t.capacity_mbps for t in type_traffic) / len(type_traffic),
                    "count": len(type_traffic),
                    "threshold_latency": self.latency_thresholds[node_type],
                    "threshold_packet_loss": self.packet_loss_thresholds[node_type],
                }

        bandwidth_trends = await self._calculate_trends(traffic_data, "bandwidth")
        latency_trends = await self._calculate_trends(traffic_data, "latency")
        packet_loss_trends = await self._calculate_trends(traffic_data, "packet_loss")

        latency_anomalies = await self._detect_anomalies(traffic_data, "latency")
        packet_loss_anomalies = await self._detect_anomalies(traffic_data, "packet_loss")
        switch_time_anomalies = await self._detect_anomalies(traffic_data, "switch_time")
        jitter_anomalies = await self._detect_anomalies(traffic_data, "jitter")

        bandwidth_forecasts = await self._forecast_metric(traffic_data, "bandwidth")
        latency_forecasts = await self._forecast_metric(traffic_data, "latency")

        correlations = await self._calculate_correlations(traffic_data)
        nodes_metrics = {}
        for node_id in {t.node_id for t in traffic_data}:
            node_traffic = [t for t in traffic_data if t.node_id == node_id]
            if node_traffic:
                latest = max(node_traffic, key=lambda x: x.timestamp)
                nodes_metrics[node_id] = {
                    "node_type": latest.node_type,
                    "latest_bandwidth": latest.bandwidth,
                    "latest_latency": latest.latency,
                    "latest_packet_loss": latest.packet_loss,
                    "latest_utilization": latest.bandwidth / latest.capacity_mbps,
                    "capacity": latest.capacity_mbps,
                    "bandwidth_trend": bandwidth_trends.get(node_id, 0),
                    "latency_trend": latency_trends.get(node_id, 0),
                    "packet_loss_trend": packet_loss_trends.get(node_id, 0),
                    "bandwidth_forecast": bandwidth_forecasts.get(node_id),
                    "latency_forecast": latency_forecasts.get(node_id),
                    "anomalies": len(
                        [
                            a
                            for a in latency_anomalies + packet_loss_anomalies + switch_time_anomalies
                            if a["node_id"] == node_id
                        ],
                    ),
                }
        network_status = self._evaluate_network_status(traffic_data)

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "period_minutes": minutes,
            "total_nodes": len({t.node_id for t in traffic_data}),
            "data_points": len(traffic_data),
            "avg_metrics": {
                "bandwidth_mbps": round(avg_bandwidth, 2),
                "latency_ms": round(avg_latency, 2),
                "packet_loss_percent": round(avg_packet_loss, 2),
                "switch_time_sec": round(avg_switch_time, 3),
                "jitter_ms": round(avg_jitter, 2),
                "utilization_percent": round(avg_utilization * 100, 2),
            },
            "node_types": node_type_data,
            "nodes_metrics": nodes_metrics,
            "trends": {
                "bandwidth": bandwidth_trends,
                "latency": latency_trends,
                "packet_loss": packet_loss_trends,
            },
            "anomalies": {
                "latency": latency_anomalies,
                "packet_loss": packet_loss_anomalies,
                "switch_time": switch_time_anomalies,
                "jitter": jitter_anomalies,
                "total_count": len(latency_anomalies)
                + len(packet_loss_anomalies)
                + len(switch_time_anomalies)
                + len(jitter_anomalies),
            },
            "forecasts": {
                "bandwidth": bandwidth_forecasts,
                "latency": latency_forecasts,
            },
            "correlations": correlations,
            "network_status": network_status,
        }

    def _evaluate_network_status(self, traffic_data: Sequence[Traffic]) -> dict:
        """Оценивает общее состояние сети на основе метрик трафика.

        Args:
            traffic_data: Данные о трафике.

        Returns:
            Словарь с оценкой состояния сети.
        """
        if not traffic_data:
            return {
                "status": NetworkStatus.DEGRADED,
                "message": "Нет данных для анализа",
                "score": 0,
            }
        node_scores = {}
        for node_id in {t.node_id for t in traffic_data}:
            node_traffic = sorted(
                [t for t in traffic_data if t.node_id == node_id],
                key=lambda x: x.timestamp,
            )
            if not node_traffic:
                continue

            latest = node_traffic[-1]
            node_type = latest.node_type
            latency_score = min(
                1.0,
                self.latency_thresholds[node_type] / max(latest.latency, 1),
            )
            packet_loss_score = min(
                1.0,
                self.packet_loss_thresholds[node_type] / max(latest.packet_loss, 0.1),
            )
            utilization_score = 1.0 - min(1.0, latest.bandwidth / latest.capacity_mbps)
            switch_score = 1.0 - min(1.0, latest.switch_time / (self.switch_time_threshold * 2))
            node_scores[node_id] = {
                "score": round(
                    (latency_score * 0.3 + packet_loss_score * 0.3 + utilization_score * 0.25 + switch_score * 0.15)
                    * 100,
                    1,
                ),
                "type": node_type,
            }
        avg_score = 0 if not node_scores else sum(n["score"] for n in node_scores.values()) / len(node_scores)
        if avg_score >= 80:
            status = NetworkStatus.OPTIMAL
            message = "Сеть функционирует оптимально"
        elif avg_score >= 60:
            status = NetworkStatus.DEGRADED
            message = "Незначительное ухудшение производительности сети"
        elif avg_score >= 40:
            status = NetworkStatus.CRITICAL
            message = "Критическое состояние сети, требуется оптимизация"
        else:
            status = NetworkStatus.CRITICAL
            message = "Серьезные проблемы с сетью, нужны экстренные меры"

        return {
            "status": status,
            "message": message,
            "score": round(avg_score, 1),
            "node_scores": node_scores,
        }

    async def get_detailed_node_analytics(self, node_id: str, minutes: int = 60) -> dict:
        """Возвращает детальную аналитику по конкретному узлу сети.

        Args:
            node_id: Идентификатор узла.
            minutes: Период анализа в минутах.

        Returns:
            Словарь с детальной аналитикой по узлу.
        """
        query = select(Traffic).filter(
            Traffic.node_id == node_id,
            Traffic.timestamp >= datetime.now(UTC) - timedelta(minutes=minutes),
        )
        result = await self.db.scalars(query)
        traffic_data = result.all()

        if not traffic_data:
            return {"message": f"Узел {node_id} не найден или нет данных"}
        traffic_data = sorted(traffic_data, key=lambda x: x.timestamp)
        latest = traffic_data[-1]
        node_type = latest.node_type
        bandwidth_values = [t.bandwidth for t in traffic_data]
        latency_values = [t.latency for t in traffic_data]
        packet_loss_values = [t.packet_loss for t in traffic_data]
        utilization_values = [t.bandwidth / t.capacity_mbps for t in traffic_data]
        jitter_values = [getattr(t, "jitter", 0) for t in traffic_data]
        [t.switch_time for t in traffic_data]
        timestamps = [t.timestamp.isoformat() for t in traffic_data]
        avg_bandwidth = sum(bandwidth_values) / len(bandwidth_values)
        avg_latency = sum(latency_values) / len(latency_values)
        avg_packet_loss = sum(packet_loss_values) / len(packet_loss_values)
        avg_jitter = sum(jitter_values) / len(traffic_data)
        avg_utilization = sum(utilization_values) / len(utilization_values)
        max_utilization = max(utilization_values)
        std_bandwidth = float(np.std(bandwidth_values)) if len(bandwidth_values) > 1 else 0
        std_latency = float(np.std(latency_values)) if len(latency_values) > 1 else 0
        std_packet_loss = float(np.std(packet_loss_values)) if len(packet_loss_values) > 1 else 0
        std_jitter = float(np.std(jitter_values)) if len(jitter_values) > 1 else 0
        bandwidth_trend = 0
        latency_trend = 0
        packet_loss_trend = 0
        if len(traffic_data) >= 5:
            try:
                recent_data = traffic_data[-5:]
                x = list(range(len(recent_data)))
                bandwidth_trend, _ = np.polyfit(x, [t.bandwidth for t in recent_data], 1)
                latency_trend, _ = np.polyfit(x, [t.latency for t in recent_data], 1)
                packet_loss_trend, _ = np.polyfit(x, [t.packet_loss for t in recent_data], 1)
            except Exception as e:
                logger.warning(f"Ошибка расчета тренда: {e}")
        bandwidth_forecast = avg_bandwidth
        latency_forecast = avg_latency
        if len(traffic_data) >= 10:
            try:
                bw_forecast = await self._forecast_metric([latest], "bandwidth")
                lat_forecast = await self._forecast_metric([latest], "latency")
                bandwidth_forecast = bw_forecast.get(node_id, avg_bandwidth)
                latency_forecast = lat_forecast.get(node_id, avg_latency)
            except Exception as e:
                logger.warning(f"Ошибка прогнозирования: {e}")
        switch_events = [
            {
                "timestamp": t.timestamp.isoformat(),
                "from_node": t.switched_from,
                "reason": t.switch_reason,
                "time_sec": t.switch_time,
                "packet_loss_percent": t.switch_packet_loss,
            }
            for t in traffic_data
            if t.switch_time > 0 and t.switched_from
        ]
        is_latency_anomaly = latest.latency > self.latency_thresholds[node_type] * self.anomaly_sensitivity
        is_packet_loss_anomaly = latest.packet_loss > self.packet_loss_thresholds[node_type] * self.anomaly_sensitivity
        is_utilization_anomaly = latest.bandwidth / latest.capacity_mbps > self.utilization_threshold_high
        is_jitter_anomaly = getattr(latest, "jitter", 0) > 100 * self.anomaly_sensitivity
        node_score = self._calculate_node_score(latest)
        if node_score["score"] >= 80:
            status = NetworkStatus.OPTIMAL
        elif node_score["score"] >= 60:
            status = NetworkStatus.DEGRADED
        elif node_score["score"] >= 40:
            status = NetworkStatus.CRITICAL
        else:
            status = NetworkStatus.CRITICAL
        optimization_suggestions = self._generate_node_optimization_suggestions(
            node_id,
            node_type,
            latest,
            is_latency_anomaly,
            is_packet_loss_anomaly,
            is_utilization_anomaly,
        )

        return {
            "node_id": node_id,
            "node_type": node_type,
            "status": str(status),
            "score": node_score["score"],
            "data_points": len(traffic_data),
            "period_minutes": minutes,
            "current_metrics": {
                "bandwidth_mbps": round(latest.bandwidth, 2),
                "latency_ms": round(latest.latency, 2),
                "packet_loss_percent": round(latest.packet_loss, 2),
                "jitter_ms": round(getattr(latest, "jitter", 0), 2),
                "utilization_percent": round(latest.bandwidth / latest.capacity_mbps * 100, 2),
                "capacity_mbps": round(latest.capacity_mbps, 2),
                "timestamp": latest.timestamp.isoformat(),
            },
            "average_metrics": {
                "bandwidth_mbps": round(avg_bandwidth, 2),
                "latency_ms": round(avg_latency, 2),
                "packet_loss_percent": round(avg_packet_loss, 2),
                "jitter_ms": round(avg_jitter, 2),
                "utilization_percent": round(avg_utilization * 100, 2),
                "max_utilization_percent": round(max_utilization * 100, 2),
            },
            "variability": {
                "bandwidth_std": round(std_bandwidth, 2),
                "latency_std": round(std_latency, 2),
                "packet_loss_std": round(std_packet_loss, 2),
                "jitter_std": round(std_jitter, 2),
            },
            "trends": {
                "bandwidth": round(float(bandwidth_trend), 4),
                "latency": round(float(latency_trend), 4),
                "packet_loss": round(float(packet_loss_trend), 4),
            },
            "forecasts": {
                "bandwidth_mbps": round(bandwidth_forecast, 2),
                "latency_ms": round(latency_forecast, 2),
            },
            "time_series": {
                "timestamps": timestamps,
                "bandwidth_mbps": [round(bw, 2) for bw in bandwidth_values],
                "latency_ms": [round(lat, 2) for lat in latency_values],
                "packet_loss_percent": [round(pl, 2) for pl in packet_loss_values],
                "utilization_percent": [round(util * 100, 2) for util in utilization_values],
            },
            "anomalies": {
                "latency": is_latency_anomaly,
                "packet_loss": is_packet_loss_anomaly,
                "utilization": is_utilization_anomaly,
                "jitter": is_jitter_anomaly,
            },
            "switch_events": switch_events,
            "optimization_suggestions": optimization_suggestions,
            "thresholds": {
                "latency_ms": self.latency_thresholds[node_type],
                "packet_loss_percent": self.packet_loss_thresholds[node_type],
                "high_utilization_percent": round(self.utilization_threshold_high * 100, 2),
                "low_utilization_percent": round(self.utilization_threshold_low * 100, 2),
            },
        }

    def _calculate_node_score(self, node_traffic: Traffic) -> dict:
        """Рассчитывает оценку производительности узла.

        Args:
            node_traffic: Данные о трафике узла.

        Returns:
            Словарь с оценкой и составляющими.
        """
        node_type = node_traffic.node_type
        latency_score = min(1.0, self.latency_thresholds[node_type] / max(node_traffic.latency, 1))
        packet_loss_score = min(1.0, self.packet_loss_thresholds[node_type] / max(node_traffic.packet_loss, 0.1))
        utilization_score = 1.0 - min(1.0, node_traffic.bandwidth / node_traffic.capacity_mbps)
        switch_score = 1.0 - min(1.0, node_traffic.switch_time / (self.switch_time_threshold * 2))
        score = (latency_score * 0.3 + packet_loss_score * 0.3 + utilization_score * 0.25 + switch_score * 0.15) * 100

        return {
            "score": round(score, 1),
            "components": {
                "latency": round(latency_score * 100, 1),
                "packet_loss": round(packet_loss_score * 100, 1),
                "utilization": round(utilization_score * 100, 1),
                "switch_time": round(switch_score * 100, 1),
            },
        }

    def _generate_node_optimization_suggestions(
        self,
        node_id: str,
        node_type: str,
        latest: Traffic,
        is_latency_anomaly: bool,
        is_packet_loss_anomaly: bool,
        is_utilization_anomaly: bool,
    ) -> list[dict]:
        """Генерирует предложения по оптимизации узла.

        Args:
            node_id: ID узла.
            node_type: Тип узла.
            latest: Последние данные по трафику.
            is_latency_anomaly: Флаг аномалии задержки.
            is_packet_loss_anomaly: Флаг аномалии потери пакетов.
            is_utilization_anomaly: Флаг аномалии утилизации.

        Returns:
            Список предложений по оптимизации.
        """
        suggestions = []
        utilization = latest.bandwidth / latest.capacity_mbps
        if is_utilization_anomaly or utilization > self.utilization_threshold_high:
            suggestions.append(
                {
                    "type": "load_balancing",
                    "priority": "high",
                    "description": f"Перенаправить часть трафика с перегруженного узла {node_id} (загрузка: {utilization:.2%})",
                    "estimated_improvement": "10-30% снижение загрузки",
                    "implementation": "Настроить балансировщик для перенаправления трафика на менее загруженные узлы",
                },
            )
        if is_latency_anomaly:
            suggestions.append(
                {
                    "type": "qos_config",
                    "priority": "medium",
                    "description": f"Оптимизировать QoS для узла {node_id} с высокой задержкой ({latest.latency:.2f} мс)",
                    "estimated_improvement": "20-40% снижение задержки для критического трафика",
                    "implementation": "Приоритизировать чувствительный к задержке трафик через механизмы QoS",
                },
            )
        if is_packet_loss_anomaly:
            suggestions.append(
                {
                    "type": "error_correction",
                    "priority": "high",
                    "description": f"Включить механизмы коррекции ошибок для узла {node_id} ({latest.packet_loss:.2f}%)",
                    "estimated_improvement": "50-70% снижение потери пакетов",
                    "implementation": "Настроить FEC (Forward Error Correction) или повторную отправку пакетов",
                },
            )
        if node_type == "satellite" and latest.latency > 300:
            suggestions.append(
                {
                    "type": "cache_config",
                    "priority": "medium",
                    "description": "Настроить локальное кэширование для снижения влияния высокой задержки",
                    "estimated_improvement": "30-50% улучшение пользовательского опыта",
                    "implementation": "Добавить прокси-сервер с кэшированием часто запрашиваемого контента",
                },
            )
        if node_type == "5G" and is_latency_anomaly:
            suggestions.append(
                {
                    "type": "hybrid_routing",
                    "priority": "medium",
                    "description": "Использовать гибридную маршрутизацию для чувствительного к задержке трафика",
                    "estimated_improvement": "20-30% снижение задержки для VoIP/видео",
                    "implementation": "Настроить правила маршрутизации для перенаправления критичного трафика через fiber-узлы",
                },
            )

        return suggestions

    async def get_network_topology(self) -> dict:
        """Получает топологию сети с информацией о связях между узлами.

        Returns:
            Словарь с описанием топологии сети.
        """
        query = select(Traffic).distinct(Traffic.node_id).order_by(Traffic.node_id, Traffic.timestamp.desc())
        nodes_result = await self.db.scalars(query)
        nodes_data = nodes_result.all()
        links_query = select(NetworkLink).filter(NetworkLink.is_active is True)
        links_result = await self.db.scalars(links_query)
        links_data = links_result.all()
        if not links_data:
            links_data = await self._generate_network_links(nodes_data)
        nodes = []
        for node in nodes_data:
            node_score = self._calculate_node_score(node)
            nodes.append(
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "connections": [link.target_node for link in links_data if link.source_node == node.node_id],
                    "status": str(NetworkStatus.OPTIMAL)
                    if node_score["score"] > 70
                    else str(NetworkStatus.DEGRADED)
                    if node_score["score"] > 50
                    else str(NetworkStatus.CRITICAL),
                    "metrics": {
                        "bandwidth": round(node.bandwidth, 2),
                        "latency": round(node.latency, 2),
                        "packet_loss": round(node.packet_loss, 2),
                        "utilization": round(node.bandwidth / node.capacity_mbps, 2),
                        "score": node_score["score"],
                    },
                },
            )
        links = []
        for link in links_data:
            links.append(
                {
                    "source": link.source_node,
                    "target": link.target_node,
                    "bandwidth": round(link.bandwidth, 2),
                    "latency": round(link.latency, 2),
                    "type": link.link_type,
                    "weight": link.weight,
                },
            )

        return {
            "nodes": nodes,
            "links": links,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def _generate_network_links(self, nodes_data: list[Traffic]) -> list:
        """Генерирует связи между узлами на основе данных о трафике.

        Args:
            nodes_data: Данные о трафике на узлах.

        Returns:
            Список сгенерированных связей.
        """
        links = []
        [node.node_id for node in nodes_data]
        for node in nodes_data:
            same_type_nodes = [n for n in nodes_data if n.node_type == node.node_type and n.node_id != node.node_id]
            for target in same_type_nodes[:2]:
                links.append(
                    NetworkLink(
                        source_node=node.node_id,
                        target_node=target.node_id,
                        bandwidth=min(node.capacity_mbps, target.capacity_mbps) * 0.7,
                        latency=(node.latency + target.latency) * 0.5,
                        is_active=True,
                        link_type=f"{node.node_type}-{target.node_type}",
                        weight=1.0,
                    ),
                )
            other_type_nodes = [n for n in nodes_data if n.node_type != node.node_type]
            for target in other_type_nodes[:1]:
                links.append(
                    NetworkLink(
                        source_node=node.node_id,
                        target_node=target.node_id,
                        bandwidth=min(node.capacity_mbps, target.capacity_mbps) * 0.5,
                        latency=(node.latency + target.latency) * 0.7,
                        is_active=True,
                        link_type=f"{node.node_type}-{target.node_type}",
                        weight=1.5,
                    ),
                )

        return links
