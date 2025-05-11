import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Traffic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficOptimizer:
    """Класс для оптимизации трафика в гибридных сетях с поддержкой автоматической настройки."""

    def __init__(self, db: AsyncSession):
        self.db = db
        
        self.latency_thresholds = {"fiber": 50, "satellite": 700, "5G": 50}
        
        self.switch_time_threshold = 0.3
        
        self.utilization_threshold_high = 0.8
        self.utilization_threshold_low = 0.5
        
        self.weights = {"fiber": 1.0, "5G": 0.8, "satellite": 0.5}

    async def _fetch_traffic_data(self, minutes: int = 10) -> list[Traffic]:
        """Получает данные о трафике за последние N минут."""
        query = select(Traffic).filter(Traffic.timestamp >= datetime.now(UTC) - timedelta(minutes=minutes))
        result = await self.db.scalars(query)
        return result.all()

    def _calculate_node_weight(self, node: Traffic) -> float:
        """Рассчитывает вес узла для балансировки на основе типа, задержки и загрузки."""
        base_weight = self.weights.get(node.node_type, 1.0)
        latency_factor = max(1.0, node.latency / self.latency_thresholds[node.node_type])
        utilization = node.bandwidth / node.capacity_mbps
        return base_weight / (latency_factor * (1 + utilization))

    async def _analyze_nodes(self, traffic_data: list[Traffic]) -> dict[str, list[Traffic]]:
        """Группирует узлы по типам и анализирует их состояние."""
        nodes_by_type = {"fiber": [], "satellite": [], "5G": []}
        for t in traffic_data:
            nodes_by_type[t.node_type].append(t)
        return nodes_by_type

    async def _generate_load_balancing_recommendations(
        self, nodes_by_type: dict[str, list[Traffic]]
    ) -> list[dict]:
        """Генерирует рекомендации по балансировке нагрузки с использованием Weighted Round-Robin."""
        recommendations = []

        for node_type, nodes in nodes_by_type.items():
            if not nodes:
                continue

            
            node_stats = [
                (t.node_id, t.bandwidth / t.capacity_mbps, self._calculate_node_weight(t), t)
                for t in nodes
            ]
            overloaded = [
                (node_id, util, weight, t)
                for node_id, util, weight, t in node_stats
                if util > self.utilization_threshold_high
            ]
            underloaded = [
                (node_id, util, weight, t)
                for node_id, util, weight, t in node_stats
                if util < self.utilization_threshold_low
            ]

            
            for overload_node, overload_util, _, node_data in overloaded:
                if underloaded:
                    
                    best_underloaded = max(underloaded, key=lambda x: x[2])
                    underload_node, underload_util, _, _ = best_underloaded
                    excess_traffic = (overload_util - self.utilization_threshold_high) * node_data.capacity_mbps

                    recommendations.append(
                        {
                            "action": "Перенаправить трафик",
                            "from_node": overload_node,
                            "to_node": underload_node,
                            "amount_mbps": round(excess_traffic, 2),
                            "reason": f"Узел {overload_node} перегружен (загрузка: {overload_util:.2f}), "
                                      f"перенаправить на {underload_node} (загрузка: {underload_util:.2f})",
                            "automation": "SDN: перенаправить через OpenFlow или API"
                        }
                    )

        return recommendations

    async def _generate_qos_recommendations(
        self, nodes_by_type: dict[str, list[Traffic]]
    ) -> list[dict]:
        """Генерирует рекомендации по приоритизации критического трафика."""
        recommendations = []

        critical_traffic_nodes = []
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                if (node_type != "satellite" and
                    node.latency < self.latency_thresholds[node_type]):
                    critical_traffic_nodes.append(node.node_id)
                    recommendations.append(
                        {
                            "action": "Приоритизировать критический трафик",
                            "node": node.node_id,
                            "reason": f"Узел {node.node_id} имеет низкую задержку "
                                      f"({node.latency} мс < {self.latency_thresholds[node_type]} мс)",
                            "automation": "SDN: установить высокий приоритет через DiffServ или QoS-политику"
                        }
                    )

        if critical_traffic_nodes:
            recommendations.append(
                {
                    "action": "Групповая приоритизация",
                    "nodes": critical_traffic_nodes,
                    "reason": "Подходит для критических приложений (VoIP, видео)",
                    "automation": "SDN: применить групповую политику QoS"
                }
            )

        return recommendations

    async def _generate_switch_reduction_recommendations(
        self, nodes_by_type: dict[str, list[Traffic]]
    ) -> list[dict]:
        """Генерирует рекомендации по минимизации переключений."""
        recommendations = []

        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                if node.switch_time > self.switch_time_threshold:
                    recommendations.append(
                        {
                            "action": "Сократить переключения",
                            "node": node.node_id,
                            "reason": f"Высокое время переключения ({node.switch_time:.2f} с > "
                                      f"{self.switch_time_threshold} с)",
                            "automation": "SDN: закрепить трафик на узле через статический маршрут"
                        }
                    )

        return recommendations

    async def optimize_traffic(self) -> dict[str, list[dict]]:
        """Генерирует полный набор рекомендаций по оптимизации трафика."""
        traffic_data = await self._fetch_traffic_data()
        if not traffic_data:
            return {"message": "Нет данных для оптимизации", "recommendations": []}

        nodes_by_type = await self._analyze_nodes(traffic_data)

        recommendations = []
        recommendations.extend(await self._generate_load_balancing_recommendations(nodes_by_type))
        recommendations.extend(await self._generate_qos_recommendations(nodes_by_type))
        recommendations.extend(await self._generate_switch_reduction_recommendations(nodes_by_type))

        if recommendations:
            recommendations.append(
                {
                    "action": "Автоматизировать оптимизацию",
                    "reason": "Использовать SDN для автоматического управления трафиком",
                    "automation": "Интеграция с OpenFlow или REST API для применения всех рекомендаций"
                }
            )

        return {
            "message": "Рекомендации по оптимизации трафика в гибридных сетях",
            "recommendations": recommendations
        }

    async def apply_optimization(self, recommendation: dict) -> dict[str, str]:
        """Применяет выбранную рекомендацию (заглушка для SDN-интеграции)."""
        return {
            "message": f"Рекомендация '{recommendation['action']}' успешно применена",
            "details": recommendation.get("automation", "Ручное применение")
        }

    async def get_optimization_status(self) -> dict[str, str]:
        """Возвращает текущий статус оптимизации."""
        traffic_data = await self._fetch_traffic_data(minutes=5)
        if not traffic_data:
            return {"status": "Нет данных для анализа"}

        total_nodes = len(traffic_data)
        overloaded_nodes = sum(
            1 for t in traffic_data if t.bandwidth / t.capacity_mbps > self.utilization_threshold_high
        )
        high_switch_nodes = sum(1 for t in traffic_data if t.switch_time > self.switch_time_threshold)

        return {
            "status": "Анализ завершен",
            "details": f"Всего узлов: {total_nodes}, Перегружено: {overloaded_nodes}, "
                       f"Высокое время переключения: {high_switch_nodes}"
        }
