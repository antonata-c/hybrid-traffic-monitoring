import logging
from datetime import UTC, datetime, timedelta
import re
import uuid

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from enums import NetworkStatus, TrafficType
from models import OptimizationAction, QoSPolicy, Traffic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficOptimizer:
    """Класс для оптимизации трафика в гибридных сетях с поддержкой автоматической настройки."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.latency_thresholds = {
            "fiber": settings.LATENCY_THRESHOLD_FIBER, 
            "satellite": settings.LATENCY_THRESHOLD_SATELLITE, 
            "5G": settings.LATENCY_THRESHOLD_5G,
            "microwave": 50,
            "starlink": 200,
            "hybrid": 100
        }
        self.packet_loss_thresholds = {
            "fiber": 1, 
            "satellite": 3, 
            "5G": 2,
            "microwave": 2,
            "starlink": 3,
            "hybrid": 2
        }
        self.switch_time_threshold = 0.3
        self.utilization_threshold_high = settings.HIGH_UTILIZATION_THRESHOLD / 100.0
        self.utilization_threshold_low = settings.LOW_UTILIZATION_THRESHOLD / 100.0
        self.weights = {
            "fiber": 1.0,
            "5G": 0.8,
            "satellite": 0.5,
            "microwave": 0.7,
            "starlink": 0.6,
            "hybrid": 0.9
        }
        self.traffic_priorities = {
            TrafficType.VOICE: 1,  
            TrafficType.VIDEO: 2,
            TrafficType.INTERACTIVE: 3,
            TrafficType.STREAMING: 4,
            TrafficType.DATA: 5,
            TrafficType.IOT: 3,
            TrafficType.SIGNALING: 1
        }
        self.route_stability_time = settings.ROUTE_STABILITY_TIME_SECONDS
        self.load_balancer_algorithm = settings.LOAD_BALANCER_ALGORITHM
        self.traffic_shaping_enabled = settings.TRAFFIC_SHAPING_ENABLED
        self.congestion_control_algorithm = settings.CONGESTION_CONTROL_ALGORITHM

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
        stability_factor = 1.0 + (node.switch_time / self.switch_time_threshold)
        return base_weight / (latency_factor * stability_factor * (1 + utilization))

    async def _analyze_nodes(self, traffic_data: list[Traffic]) -> dict[str, list[Traffic]]:
        """Группирует узлы по типам и анализирует их состояние."""
        nodes_by_type = {"fiber": [], "satellite": [], "5G": []}
        for t in traffic_data:
            nodes_by_type[t.node_type].append(t)
        for node_type, nodes in nodes_by_type.items():
            if nodes:
                avg_utilization = sum(n.bandwidth / n.capacity_mbps for n in nodes) / len(nodes)
                avg_latency = sum(n.latency for n in nodes) / len(nodes)
                logger.info(f"Тип узла {node_type}: {len(nodes)} узлов, средняя утилизация: {avg_utilization:.2f}, "
                           f"средняя задержка: {avg_latency:.2f} мс")
            else:
                logger.warning(f"Нет активных узлов типа {node_type}")
                
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
                    underload_node, underload_util, _, underload_data = best_underloaded
                    excess_traffic = (overload_util - self.utilization_threshold_high) * node_data.capacity_mbps
                    available_capacity = (self.utilization_threshold_high - underload_util) * underload_data.capacity_mbps
                    transfer_amount = min(excess_traffic, available_capacity)
                    recommendations.append(
                        {
                            "action": "Перенаправить трафик",
                            "from_node": overload_node,
                            "to_node": underload_node,
                            "amount_mbps": round(transfer_amount, 2),
                            "reason": f"Узел {overload_node} перегружен (загрузка: {overload_util:.2f}), "
                                      f"перенаправить на {underload_node} (загрузка: {underload_util:.2f})",
                            "automation": "SDN: перенаправить через OpenFlow или API",
                            "implementation_steps": [
                                "1. Установить правило маршрутизации в SDN-контроллере",
                                f"2. Ограничить пропускную способность для узла {overload_node}",
                                f"3. Увеличить квоту для узла {underload_node}",
                                "4. Мониторить изменения в течение 5 минут"
                            ]
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
                if node.latency < self.latency_thresholds[node_type] * 0.7:  
                    critical_traffic_nodes.append(node.node_id)
                    recommendations.append(
                        {
                            "action": "Приоритизировать критический трафик",
                            "node": node.node_id,
                            "reason": f"Узел {node.node_id} имеет низкую задержку "
                                      f"({node.latency} мс < {self.latency_thresholds[node_type] * 0.7} мс)",
                            "automation": "SDN: установить высокий приоритет через DiffServ или QoS-политику",
                            "traffic_types": ["voice", "video", "interactive"],
                            "qos_settings": {
                                "dscp": "EF (Expedited Forwarding)",
                                "queue_priority": "High",
                                "bandwidth_guarantee": f"{min(50, int(node.capacity_mbps * 0.3))} Mbps"
                            }
                        }
                    )
                elif node.latency < self.latency_thresholds[node_type]:
                    recommendations.append(
                        {
                            "action": "Оптимизировать интерактивный трафик",
                            "node": node.node_id,
                            "reason": f"Узел {node.node_id} имеет среднюю задержку "
                                      f"({node.latency} мс < {self.latency_thresholds[node_type]} мс)",
                            "automation": "SDN: установить средний приоритет для интерактивных приложений",
                            "traffic_types": ["interactive", "streaming"],
                            "qos_settings": {
                                "dscp": "AF31/AF32 (Assured Forwarding)",
                                "queue_priority": "Medium",
                                "bandwidth_allocation": "30%"
                            }
                        }
                    )
        if critical_traffic_nodes:
            recommendations.append(
                {
                    "action": "Групповая приоритизация критического трафика",
                    "nodes": critical_traffic_nodes,
                    "reason": "Подходит для критических приложений (VoIP, видеоконференции, телемедицина)",
                    "automation": "SDN: применить групповую политику QoS",
                    "implementation": {
                        "policy_name": "critical_traffic_policy",
                        "traffic_classification": {
                            "voice": {"ports": [5060, 5061, 16384-32767], "protocols": ["UDP", "RTP"]},
                            "video": {"ports": [554, 8554, 1935], "protocols": ["TCP", "UDP", "RTSP", "RTMP"]}
                        },
                        "monitoring": "Настроить алерты при превышении задержки более 100 мс"
                    }
                }
            )

        return recommendations

    async def _generate_switch_reduction_recommendations(
        self, nodes_by_type: dict[str, list[Traffic]]
    ) -> list[dict]:
        """Генерирует рекомендации по минимизации переключений между узлами."""
        recommendations = []
        for node_type, nodes in nodes_by_type.items():
            high_switch_nodes = []
            
            for node in nodes:
                if node.switch_time > self.switch_time_threshold:
                    high_switch_nodes.append(node.node_id)
                    optimal_stability = max(
                        self.route_stability_time,
                        int(node.switch_time * 10)  
                    )
                    
                    recommendations.append(
                        {
                            "action": "Сократить переключения",
                            "node": node.node_id,
                            "reason": f"Высокое время переключения ({node.switch_time:.2f} с > "
                                      f"{self.switch_time_threshold} с)",
                            "automation": "SDN: закрепить трафик на узле через статический маршрут",
                            "implementation": {
                                "sticky_timeout": f"{optimal_stability} секунд",
                                "hysteresis": f"{int(node.switch_time * 2)} секунд",
                                "min_utilization_for_switch": f"{self.utilization_threshold_high + 0.1:.2f}"
                            }
                        }
                    )
            if len(high_switch_nodes) > 2:
                recommendations.append(
                    {
                        "action": "Групповая стабилизация маршрутов",
                        "nodes": high_switch_nodes,
                        "reason": f"Множественные узлы типа {node_type} с высоким временем переключения",
                        "automation": "SDN: применить политику стабильности маршрутов",
                        "implementation": {
                            "policy_name": f"stability_policy_{node_type}",
                            "min_route_age": f"{self.route_stability_time} секунд",
                            "load_threshold_for_reroute": f"{self.utilization_threshold_high + 0.15:.2f}"
                        }
                    }
                )

        return recommendations

    async def _generate_capacity_planning_recommendations(
        self, nodes_by_type: dict[str, list[Traffic]]
    ) -> list[dict]:
        """Генерирует рекомендации по планированию емкости сети."""
        recommendations = []
        
        for node_type, nodes in nodes_by_type.items():
            if not nodes:
                continue
            high_utilization_nodes = [
                node for node in nodes 
                if node.bandwidth / node.capacity_mbps > self.utilization_threshold_high * 0.9
            ]
            if high_utilization_nodes and len(high_utilization_nodes) / len(nodes) > 0.3:
                avg_utilization = sum(n.bandwidth / n.capacity_mbps for n in high_utilization_nodes) / len(high_utilization_nodes)
                
                recommendations.append({
                    "action": "Увеличить емкость сети",
                    "node_type": node_type,
                    "affected_nodes": [n.node_id for n in high_utilization_nodes],
                    "reason": f"{len(high_utilization_nodes)} узлов типа {node_type} имеют высокую утилизацию "
                              f"(в среднем {avg_utilization:.2f})",
                    "recommendation": f"Увеличить емкость на {int((avg_utilization - 0.6) * 100)}% или добавить "
                                      f"{max(1, len(high_utilization_nodes) // 3)} новых узлов",
                    "priority": "Высокий" if avg_utilization > 0.9 else "Средний"
                })
        
        return recommendations

    async def optimize_traffic(self) -> dict:
        """Генерирует рекомендации по оптимизации трафика в гибридной сети.
        
        Returns:
            Рекомендации для оптимизации трафика.
        """
        traffic_data = await self._fetch_traffic_data(minutes=settings.ANALYSIS_WINDOW_MINUTES)
        
        if not traffic_data:
            return {
                "message": "Недостаточно данных для оптимизации",
                "recommendations": [],
                "network_status": {
                    "status": NetworkStatus.DEGRADED,
                    "reason": "Нет данных о трафике"
                }
            }
        nodes_by_type = await self._analyze_nodes(traffic_data)
        load_balancing_recs = await self._generate_load_balancing_recommendations(nodes_by_type)
        qos_recs = await self._generate_qos_recommendations(nodes_by_type)
        switch_reduction_recs = await self._generate_switch_reduction_recommendations(nodes_by_type)
        capacity_planning_recs = await self._generate_capacity_planning_recommendations(nodes_by_type)
        hybrid_routing_recs = await self._generate_hybrid_routing_recommendations(nodes_by_type)
        traffic_shaping_recs = await self._generate_traffic_shaping_recommendations(nodes_by_type)
        failover_recs = await self._generate_failover_recommendations(nodes_by_type)
        all_recommendations = (
            load_balancing_recs +
            qos_recs +
            switch_reduction_recs +
            capacity_planning_recs +
            hybrid_routing_recs +
            traffic_shaping_recs +
            failover_recs
        )
        network_status = await self._calculate_network_status(traffic_data)
        all_recommendations.sort(
            key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3)
        )
        if settings.AUTO_OPTIMIZATION and all_recommendations:
            await self._store_optimization_recommendations(all_recommendations)
            
        return {
            "message": f"Сгенерировано {len(all_recommendations)} рекомендаций по оптимизации",
            "recommendations": all_recommendations,
            "network_status": network_status
        }

    async def _calculate_network_status(self, traffic_data: list) -> dict:
        """Рассчитывает текущий статус сети на основе данных о трафике.
        
        Args:
            traffic_data: Данные о трафике.
            
        Returns:
            Информация о текущем статусе сети.
        """
        total_nodes = len({t.node_id for t in traffic_data})
        node_types = {t.node_type for t in traffic_data}
        avg_utilization = sum(t.bandwidth / t.capacity_mbps for t in traffic_data) / len(traffic_data)
        overloaded_nodes = len([t for t in traffic_data if t.bandwidth / t.capacity_mbps > self.utilization_threshold_high])
        high_latency_nodes = len([
            t for t in traffic_data if t.latency > self.latency_thresholds[t.node_type]
        ])
        high_packet_loss_nodes = len([
            t for t in traffic_data if t.packet_loss > self.packet_loss_thresholds[t.node_type]
        ])
        high_switch_nodes = len([t for t in traffic_data if t.switch_time > self.switch_time_threshold])
        latency_by_type = {}
        for node_type in node_types:
            type_data = [t for t in traffic_data if t.node_type == node_type]
            if type_data:
                latency_by_type[node_type] = sum(t.latency for t in type_data) / len(type_data)
        network_utilization = avg_utilization
        if overloaded_nodes > total_nodes * 0.3:
            network_status = NetworkStatus.CRITICAL
            status_reason = "Перегрузка сети"
        elif high_latency_nodes > total_nodes * 0.3:
            network_status = NetworkStatus.CRITICAL
            status_reason = "Высокие задержки"
        elif high_packet_loss_nodes > total_nodes * 0.3:
            network_status = NetworkStatus.CRITICAL
            status_reason = "Высокие потери пакетов"
        elif overloaded_nodes > 0 or high_latency_nodes > 0 or high_packet_loss_nodes > 0:
            network_status = NetworkStatus.DEGRADED
            status_reason = "Частичное ухудшение производительности"
        elif high_switch_nodes > total_nodes * 0.3:
            network_status = NetworkStatus.DEGRADED
            status_reason = "Нестабильная маршрутизация"
        else:
            network_status = NetworkStatus.OPTIMAL
            status_reason = "Сеть функционирует оптимально"

        return {
            "status": str(network_status),
            "reason": status_reason,
            "timestamp": datetime.now(UTC).isoformat(),
            "network_utilization": f"{network_utilization:.2%}",
            "stats": {
                "total_nodes": total_nodes, 
                "overloaded_nodes": overloaded_nodes, 
                "high_latency_nodes": high_latency_nodes,
                "high_packet_loss_nodes": high_packet_loss_nodes,
                "high_switch_nodes": high_switch_nodes
            },
            "latency_by_type": {k: f"{v:.2f} мс" for k, v in latency_by_type.items()},
        }
        
    async def _store_optimization_recommendations(self, recommendations: list) -> None:
        """Сохраняет сгенерированные рекомендации в базу данных.
        
        Args:
            recommendations: Список рекомендаций.
        """
        for recommendation in recommendations[:5]:  
            action_type_slug = re.sub(r'[^a-zA-Z0-9-]', '', recommendation.get("action", "unknown").lower())
            slug = f"{action_type_slug}-{uuid.uuid4().hex[:8]}"
            action = OptimizationAction(
                slug=slug,
                action_type=recommendation.get("action", "unknown"),
                affected_nodes=recommendation.get("target_nodes", []) or [recommendation.get("node", "")],
                description=recommendation.get("description", ""),
                before_metrics=recommendation.get("current_metrics", {}),
                success=False,
                created_at=datetime.now(UTC)
            )
            self.db.add(action)
            await self.db.flush()
        await self.db.commit()
        
    async def _generate_hybrid_routing_recommendations(self, nodes_by_type: dict) -> list:
        """Генерирует рекомендации по гибридной маршрутизации трафика.
        
        Args:
            nodes_by_type: Узлы, сгруппированные по типам.
            
        Returns:
            Список рекомендаций по гибридной маршрутизации.
        """
        recommendations = []
        fiber_nodes = nodes_by_type.get("fiber", [])
        satellite_nodes = nodes_by_type.get("satellite", [])
        gen5_nodes = nodes_by_type.get("5G", [])
        if fiber_nodes and satellite_nodes:
            recommendations.append({
                "action": "Гибридная маршрутизация чувствительного трафика",
                "priority": "high",
                "target_nodes": [n.node_id for n in satellite_nodes],
                "description": f"Перенаправить чувствительный к задержке трафик с {len(satellite_nodes)} спутниковых узлов на оптоволоконные узлы",
                "reason": "Спутниковые узлы имеют высокую задержку, критичную для голосового и видео трафика",
                "estimated_improvement": "70-90% снижение задержки для голосового и видео трафика",
                "implementation": {
                    "strategy": "Маршрутизация на основе типа трафика",
                    "target_traffic": ["VoIP", "Video Conferencing", "Interactive Gaming"],
                    "commands": [
                        "Настроить DPI для классификации трафика",
                        "Создать политики маршрутизации в зависимости от типа трафика",
                        "Организовать туннели для приоритетного трафика через оптоволоконные узлы"
                    ]
                }
            })
        
        if satellite_nodes and gen5_nodes:
            recommendations.append({
                "action": "Оптимизация затрат на трафик",
                "priority": "medium",
                "target_nodes": [n.node_id for n in gen5_nodes],
                "description": f"Перенаправить некритичный объемный трафик с {len(gen5_nodes)} 5G узлов на спутниковые узлы",
                "reason": "5G трафик обычно стоит дороже спутникового для больших объемов данных",
                "estimated_improvement": "20-40% снижение затрат на передачу объемных данных",
                "implementation": {
                    "strategy": "Маршрутизация на основе объема и приоритета",
                    "target_traffic": ["Backup", "Updates", "Large Downloads"],
                    "conditions": "только в периоды низкой загрузки спутниковых каналов"
                }
            })
        
        return recommendations
    
    async def _generate_traffic_shaping_recommendations(self, nodes_by_type: dict) -> list:
        """Генерирует рекомендации по шейпингу трафика.
        
        Args:
            nodes_by_type: Узлы, сгруппированные по типам.
            
        Returns:
            Список рекомендаций по шейпингу трафика.
        """
        recommendations = []
        
        shaping_recommendations = {
            "satellite": {
                "action": "Шейпинг спутникового трафика",
                "priority": "high",
                "description": lambda nodes: f"Настроить приоритизацию трафика на {len(nodes)} перегруженных спутниковых узлах",
                "reason": "Из-за высокой задержки спутниковых каналов важно правильно приоритизировать трафик",
                "estimated_improvement": "30-50% улучшение пользовательского опыта для интерактивных приложений",
                "implementation": {
                    "strategy": "Многоуровневая очередь с приоритетами",
                    "classes": [
                        {"name": "interactive", "bandwidth": "30%", "priority": "high"},
                        {"name": "web_browsing", "bandwidth": "40%", "priority": "medium"},
                        {"name": "bulk_download", "bandwidth": "30%", "priority": "low"}
                    ],
                    "commands": [
                        "tc qdisc add dev eth0 root handle 1: htb default 30",
                        "tc class add dev eth0 parent 1: classid 1:1 htb rate 100mbit",
                        "tc class add dev eth0 parent 1:1 classid 1:10 htb rate 30mbit ceil 90mbit prio 1",
                        "tc class add dev eth0 parent 1:1 classid 1:20 htb rate 40mbit ceil 90mbit prio 2",
                        "tc class add dev eth0 parent 1:1 classid 1:30 htb rate 30mbit ceil 90mbit prio 3"
                    ]
                }
            },
            "fiber": {
                "action": "Оптимизация QoS на оптоволоконных узлах",
                "priority": "medium",
                "description": lambda nodes: f"Настроить детальные политики QoS на {len(nodes)} оптоволоконных узлах",
                "reason": "Даже при высокой пропускной способности требуется приоритизация трафика для оптимального обслуживания",
                "estimated_improvement": "15-25% улучшение отзывчивости для бизнес-приложений",
                "implementation": {
                    "strategy": "DSCP-based QoS",
                    "configurations": [
                        {"traffic": "VoIP", "dscp": "EF (46)", "bandwidth": "10%", "queue": "priority"},
                        {"traffic": "Video", "dscp": "AF41 (34)", "bandwidth": "30%", "queue": "bandwidth"},
                        {"traffic": "Business Apps", "dscp": "AF31 (26)", "bandwidth": "40%", "queue": "bandwidth"},
                        {"traffic": "Default", "dscp": "BE (0)", "bandwidth": "20%", "queue": "fair-queue"}
                    ]
                }
            },
            "5G": {
                "action": "Динамический QoS для 5G",
                "priority": "medium",
                "description": lambda nodes: f"Внедрить динамические политики QoS на {len(nodes)} 5G узлах",
                "reason": "5G сети поддерживают динамическое управление QoS в зависимости от нагрузки",
                "estimated_improvement": "20-30% повышение эффективности использования ресурсов",
                "implementation": {
                    "strategy": "5G QoS Flows with Network Slicing",
                    "slices": [
                        {"type": "eMBB", "traffic": "Streaming, Downloads", "allocation": "dynamic"},
                        {"type": "URLLC", "traffic": "Control, IoT", "allocation": "guaranteed"},
                        {"type": "mMTC", "traffic": "IoT, Sensors", "allocation": "shared"}
                    ]
                }
            },
            "microwave": {
                "action": "Оптимизация QoS для микроволновых узлов",
                "priority": "medium",
                "description": lambda nodes: f"Настроить политики QoS на {len(nodes)} микроволновых узлах",
                "reason": "Микроволновые узлы требуют баланса между задержкой и пропускной способностью",
                "estimated_improvement": "20-25% улучшение стабильности соединения",
                "implementation": {
                    "strategy": "Приоритеты трафика по типам",
                    "configurations": [
                        {"traffic": "Критический трафик", "bandwidth": "20%", "priority": "highest"},
                        {"traffic": "Рабочие приложения", "bandwidth": "50%", "priority": "medium"},
                        {"traffic": "Прочий трафик", "bandwidth": "30%", "priority": "best-effort"}
                    ]
                }
            },
            "starlink": {
                "action": "Оптимизация Starlink соединений",
                "priority": "medium",
                "description": lambda nodes: f"Оптимизировать работу {len(nodes)} узлов Starlink",
                "reason": "Starlink требует специальных настроек для минимизации джиттера и пакетных потерь",
                "estimated_improvement": "30-40% улучшение пользовательского опыта",
                "implementation": {
                    "strategy": "Буферизация и приоритизация",
                    "buffer_size": "Адаптивный размер 100-500мс",
                    "traffic_classes": [
                        {"class": "Низкая задержка", "apps": "VoIP, Видеоконференции", "priority": "высокий"},
                        {"class": "Интерактивный", "apps": "Web, SSH, RDP", "priority": "средний"},
                        {"class": "Фоновый", "apps": "Загрузки, Обновления", "priority": "низкий"}
                    ]
                }
            },
            "hybrid": {
                "action": "Гибридная оптимизация трафика",
                "priority": "high",
                "description": lambda nodes: f"Настроить интеллектуальное распределение трафика на {len(nodes)} гибридных узлах",
                "reason": "Гибридные узлы могут оптимально маршрутизировать трафик через различные каналы",
                "estimated_improvement": "40-60% улучшение соотношения скорость/стоимость",
                "implementation": {
                    "strategy": "Распределение по приложениям",
                    "routing_rules": [
                        {"app_type": "Критические", "path": "Низкая задержка (fiber/5G)", "priority": "mandatory"},
                        {"app_type": "Потоковые", "path": "Высокая пропускная способность", "priority": "preferred"},
                        {"app_type": "Фоновые", "path": "Низкая стоимость (satellite)", "priority": "cost-effective"}
                    ]
                }
            }
        }
        
        for node_type, nodes in nodes_by_type.items():
            high_util_nodes = [
                n for n in nodes if n.bandwidth / n.capacity_mbps > self.utilization_threshold_high * 0.9
            ]
            
            if high_util_nodes and node_type in shaping_recommendations:
                target_nodes = [n.node_id for n in high_util_nodes]
                recommendation = shaping_recommendations[node_type].copy()
                
                if callable(recommendation["description"]):
                    recommendation["description"] = recommendation["description"](target_nodes)
                
                recommendation["target_nodes"] = target_nodes
                recommendations.append(recommendation)
        
        return recommendations
    
    async def _generate_failover_recommendations(self, nodes_by_type: dict) -> list:
        """Генерирует рекомендации по настройке отказоустойчивости в гибридной сети.
        
        Args:
            nodes_by_type: Узлы, сгруппированные по типам.
            
        Returns:
            Список рекомендаций по отказоустойчивости.
        """
        recommendations = []
        if sum(len(nodes) for nodes in nodes_by_type.values()) < 3:
            return recommendations
        for node_type, nodes in nodes_by_type.items():
            if len(nodes) < 2:
                continue
            main_nodes = sorted(
                nodes, 
                key=lambda n: (n.latency, n.packet_loss, -(n.capacity_mbps - n.bandwidth))
            )[:len(nodes)//2 + 1]
            backup_nodes = [n for n in nodes if n not in main_nodes]
            
            if not backup_nodes:
                continue
            recommendations.append({
                "action": f"Настройка отказоустойчивости для {node_type}",
                "priority": "high",
                "target_nodes": [n.node_id for n in main_nodes],
                "backup_nodes": [n.node_id for n in backup_nodes],
                "description": f"Настроить автоматическое переключение с {len(main_nodes)} основных узлов на {len(backup_nodes)} резервных при отказах",
                "reason": "Обеспечение непрерывности бизнес-процессов при отказе основных каналов связи",
                "estimated_improvement": "99.9% доступности сервисов",
                "implementation": {
                    "strategy": "Автоматическое переключение при отказе",
                    "monitoring": "ICMP, TCP keep-alive, BFD",
                    "switch_thresholds": {
                        "packet_loss": f"{settings.PACKET_LOSS_THRESHOLD}%",
                        "latency": f"{self.latency_thresholds[node_type] * 1.5} мс",
                        "bandwidth_drop": "50%"
                    },
                    "recovery_strategy": "Автоматический возврат на основной канал после восстановления"
                }
            })
        if len(nodes_by_type) >= 2:
            priority_order = ["fiber", "5G", "microwave", "starlink", "satellite"]
            node_types_by_priority = sorted(
                nodes_by_type.keys(),
                key=lambda t: priority_order.index(t) if t in priority_order else 999
            )
            
            primary_type = node_types_by_priority[0]
            secondary_types = node_types_by_priority[1:]
            
            if primary_type and secondary_types:
                recommendations.append({
                    "action": "Межтиповая отказоустойчивость",
                    "priority": "high",
                    "target_nodes": [n.node_id for n in nodes_by_type[primary_type]],
                    "backup_nodes": [n.node_id for type_name in secondary_types for n in nodes_by_type.get(type_name, [])],
                    "description": f"Настроить автоматическое переключение с {primary_type} на {', '.join(secondary_types)} при отказах",
                    "reason": "Гибридная сеть позволяет использовать разные технологии для обеспечения непрерывности",
                    "estimated_improvement": "100% гарантия связности для критических сервисов",
                    "implementation": {
                        "strategy": "Каскадное переключение по приоритетам технологий",
                        "priority_order": [primary_type] + secondary_types,
                        "traffic_classification": {
                            "critical": ["Голос", "Видео", "Управление", "Транзакции"],
                            "important": ["Web", "Email", "CRM"],
                            "standard": ["Updates", "Backups", "Downloads"]
                        }
                    }
                })
                
        return recommendations

    async def apply_optimization(self, recommendation: dict) -> dict:
        """Применяет выбранную рекомендацию по оптимизации трафика.

        Args:
            recommendation: Рекомендация для применения.

        Returns:
            Словарь с результатом применения.
        """

        action = recommendation.get("action", "")
        logger.info(f"Применение рекомендации: {action}")

        try:
            action_type_slug = re.sub(r'[^a-zA-Z0-9-]', '', action.lower())
            slug = f"{action_type_slug}-{uuid.uuid4().hex[:8]}"
            effective_days = np.random.randint(1, 8)
            effective_until = datetime.now(UTC) + timedelta(days=effective_days)
            optimization_record = OptimizationAction(
                slug=slug,
                action_type=action,
                affected_nodes=recommendation.get("target_nodes", []) or [recommendation.get("node", "")],
                description=recommendation.get("description", ""),
                before_metrics=recommendation.get("current_metrics", {}),
                success=True,
                created_at=datetime.now(UTC),
                effective_until=effective_until,
                is_active=True
            )
            self.db.add(optimization_record)
            await self.db.commit()
            return {
                "status": "success",
                "message": f"Рекомендация '{action}' успешно применена",
                "affected_nodes": recommendation.get("target_nodes", []) or [recommendation.get("node", "")],
                "optimization_id": slug,
                "effective_until": effective_until.isoformat(),
                "timestamp": datetime.now(UTC).isoformat()
            }
        except Exception as e:
            logger.error(f"Ошибка применения оптимизации: {e}")
        return {
                "status": "error",
                "message": f"Ошибка применения оптимизации: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat()
            }

    async def manual_route_switch(self, from_node: str, to_node: str, traffic_percentage: float) -> dict:
        """Выполняет ручное переключение трафика между узлами.
        
        Args:
            from_node: Исходный узел.
            to_node: Целевой узел.
            traffic_percentage: Процент трафика для переключения (0-100).
            
        Returns:
            Результат переключения.
        """
        logger.info(f"Ручное переключение трафика: {from_node} -> {to_node} ({traffic_percentage}%)")
        
        try:
            query = (
                select(Traffic)
                .filter(Traffic.node_id.in_([from_node, to_node]))
                .order_by(Traffic.node_id, Traffic.timestamp.desc())
            )
            result = await self.db.scalars(query)
            traffic_data = result.all()
            
            if len(traffic_data) < 2:
                return {
                    "status": "error",
                    "message": "Не найдены данные для одного или обоих узлов",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            logger.error(traffic_data[1].node_id)
            logger.error((to_node, from_node))
            source_node = next((t for t in traffic_data if t.node_id == from_node), None)
            target_node = next((t for t in traffic_data if t.node_id == to_node), None)
            
            if not source_node or not target_node:
                return {
                    "status": "error",
                    "message": "Не удалось определить параметры узлов",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            source_bandwidth = source_node.bandwidth
            traffic_to_switch = source_bandwidth * (traffic_percentage / 100.0)
            target_available = target_node.capacity_mbps - target_node.bandwidth
            action_type_slug = re.sub(r'[^a-zA-Z0-9-]', '', to_node.lower())
            slug = f"{action_type_slug}-{uuid.uuid4().hex[:8]}"
            
            if traffic_to_switch > target_available:
                return {
                    "status": "warning",
                    "message": f"Целевой узел может принять только {target_available:.2f} Mbps из запрошенных {traffic_to_switch:.2f} Mbps",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "actual_percentage": round(min(traffic_percentage, target_available / source_bandwidth * 100), 2)
                }
            switch_action = OptimizationAction(
                slug=slug,
                action_type="manual_route_switch",
                affected_nodes=[from_node, to_node],
                description=f"Ручное переключение {traffic_percentage}% трафика с {from_node} на {to_node}",
                before_metrics={
                    "source_bandwidth": source_node.bandwidth,
                    "source_utilization": source_node.bandwidth / source_node.capacity_mbps,
                    "target_bandwidth": target_node.bandwidth,
                    "target_utilization": target_node.bandwidth / target_node.capacity_mbps
                },
                success=True,
                created_at=datetime.now(UTC)
            )
            self.db.add(switch_action)
            await self.db.flush()
            await self.db.commit()

            await self.db.refresh(source_node)
            await self.db.refresh(target_node)

            return {
                    "status": "success",
                    "message": f"Переключение {traffic_percentage}% трафика с {from_node} на {to_node} выполнено успешно",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "details": {
                        "traffic_switched_mbps": round(traffic_to_switch, 2),
                        "source_node": {
                            "id": from_node,
                            "type": source_node.node_type,
                            "before_bandwidth": round(source_node.bandwidth, 2),
                            "after_bandwidth": round(source_node.bandwidth - traffic_to_switch, 2)
                        },
                        "target_node": {
                            "id": to_node,
                            "type": target_node.node_type,
                            "before_bandwidth": round(target_node.bandwidth, 2),
                            "after_bandwidth": round(target_node.bandwidth + traffic_to_switch, 2)
                        }
                    }
                }
        except Exception as e:
            logger.exception(f"Ошибка при переключении трафика: {e}")
            return {
                "status": "error",
                "message": f"Ошибка при переключении трафика: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat()
            }
            
    async def configure_qos_policies(self, node_id: str, traffic_type: str, priority: int) -> dict:
        """Настраивает QoS-политики для указанного типа трафика на узле.
        
        Args:
            node_id: Идентификатор узла.
            traffic_type: Тип трафика.
            priority: Приоритет (1-5, где 1 - наивысший).
            
        Returns:
            Результат настройки.
        """
        logger.info(f"Настройка QoS: узел {node_id}, тип {traffic_type}, приоритет {priority}")
        
        try:
            query = (
                select(Traffic)
                .filter(Traffic.node_id == node_id)
                .order_by(Traffic.timestamp.desc())
                .limit(1)
            )
            result = await self.db.scalar(query)
            
            if not result:
                return {
                    "status": "error",
                    "message": f"Узел {node_id} не найден",
                    "timestamp": datetime.now(UTC).isoformat()
                }
                
            qos_query = (
                select(QoSPolicy)
                .filter(
                    QoSPolicy.node_id == node_id,
                    QoSPolicy.traffic_type == traffic_type
                )
            )
            existing_policy = await self.db.scalar(qos_query)
                
            traffic_params = {
                TrafficType.VOICE: {
                    "bandwidth_reserved": 0.1 * result.capacity_mbps,
                    "max_latency": min(50, self.latency_thresholds[result.node_type] * 0.5),
                    "max_packet_loss": 0.5
                },
                TrafficType.VIDEO: {
                    "bandwidth_reserved": 0.2 * result.capacity_mbps,
                    "max_latency": min(100, self.latency_thresholds[result.node_type] * 0.7),
                    "max_packet_loss": 1.0
                },
                TrafficType.INTERACTIVE: {
                    "bandwidth_reserved": 0.15 * result.capacity_mbps,
                    "max_latency": min(150, self.latency_thresholds[result.node_type] * 0.8),
                    "max_packet_loss": 1.5
                },
                TrafficType.STREAMING: {
                    "bandwidth_reserved": 0.25 * result.capacity_mbps,
                    "max_latency": min(300, self.latency_thresholds[result.node_type]),
                    "max_packet_loss": 2.0
                },
                TrafficType.DATA: {
                    "bandwidth_reserved": 0.1 * result.capacity_mbps,
                    "max_latency": self.latency_thresholds[result.node_type] * 1.5,
                    "max_packet_loss": 3.0
                },
                TrafficType.IOT: {
                    "bandwidth_reserved": 0.05 * result.capacity_mbps,
                    "max_latency": min(200, self.latency_thresholds[result.node_type]),
                    "max_packet_loss": 2.0
                },
                TrafficType.SIGNALING: {
                    "bandwidth_reserved": 0.05 * result.capacity_mbps,
                    "max_latency": min(80, self.latency_thresholds[result.node_type] * 0.6),
                    "max_packet_loss": 0.1
                }
            }
            
            params = traffic_params.get(traffic_type, {
                "bandwidth_reserved": 0.1 * result.capacity_mbps,
                "max_latency": self.latency_thresholds[result.node_type],
                "max_packet_loss": 2.0
            })
            action_type_slug = re.sub(r'[^a-zA-Z0-9-]', '', node_id.lower())
            slug = f"{action_type_slug}-{uuid.uuid4().hex[:8]}"
            
            if existing_policy:
                existing_policy.priority = priority
                existing_policy.bandwidth_reserved = params["bandwidth_reserved"]
                existing_policy.max_latency = params["max_latency"]
                existing_policy.max_packet_loss = params["max_packet_loss"]
                policy_id = existing_policy.id
                message = f"QoS-политика для {traffic_type} на узле {node_id} обновлена"
            else:
                new_policy = QoSPolicy(
                    node_id=node_id,
                    traffic_type=traffic_type,
                    priority=priority,
                    bandwidth_reserved=params["bandwidth_reserved"],
                    max_latency=params["max_latency"],
                    max_packet_loss=params["max_packet_loss"],
                    description=f"QoS для {traffic_type} на узле {node_id}",
                    is_active=True,
                    created_at=datetime.now(UTC)
                )
                self.db.add(new_policy)
                await self.db.flush()
                await self.db.commit()
                await self.db.refresh(new_policy)
                policy_id = new_policy.id
                message = f"Новая QoS-политика для {traffic_type} на узле {node_id} создана"
                
            qos_action = OptimizationAction(
                slug=slug,
                action_type="qos_configuration",
                affected_nodes=[node_id],
                description=f"Настройка QoS для {traffic_type} с приоритетом {priority} на узле {node_id}",
                before_metrics={
                    "node_type": result.node_type,
                    "bandwidth": result.bandwidth,
                    "capacity": result.capacity_mbps,
                    "latency": result.latency,
                    "packet_loss": result.packet_loss
                },
                success=True,
                created_at=datetime.now(UTC)
            )
            self.db.add(qos_action)
            await self.db.flush()
            await self.db.commit()
            await self.db.refresh(result)
                
            return {
                    "status": "success",
                "message": message,
                "timestamp": datetime.now(UTC).isoformat(),
                "policy_id": policy_id,
                "node_id": node_id,
                "traffic_type": traffic_type,
                "priority": priority,
                "details": {
                    "bandwidth_reserved_mbps": round(params["bandwidth_reserved"], 2),
                    "max_latency_ms": round(params["max_latency"], 2),
                    "max_packet_loss_percent": round(params["max_packet_loss"], 2),
                    "node_capacity_mbps": round(result.capacity_mbps, 2)
                }
            }
        except Exception as e:
            logger.error(f"Ошибка настройки QoS: {e}")
            return {
                "status": "error",
                "message": f"Ошибка настройки QoS: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat()
            }
    async def get_active_optimizations(self) -> list:
        """Получает список активных оптимизаций из базы данных.
        
        Returns:
            Список активных оптимизаций.
        """
        try:
            current_time = datetime.now(UTC)
            query = (
                select(OptimizationAction)
                .filter(
                    OptimizationAction.is_active == True,
                    (OptimizationAction.effective_until == None) | (OptimizationAction.effective_until > current_time)
                )
                .order_by(desc(OptimizationAction.created_at))
            )
            result = await self.db.scalars(query)
            optimizations = result.all()
            return [
                {
                    "id": opt.id,
                    "slug": opt.slug,
                    "action_type": opt.action_type,
                    "affected_nodes": opt.affected_nodes,
                    "description": opt.description,
                    "created_at": opt.created_at.isoformat(),
                    "effective_until": opt.effective_until.isoformat() if opt.effective_until else None,
                    "days_remaining": (opt.effective_until - current_time).days if opt.effective_until else None
                }
                for opt in optimizations
            ]
        except Exception as e:
            logger.error(f"Ошибка получения активных оптимизаций: {e}")
            return []

    async def update_optimization_statuses(self) -> dict:
        """Обновляет статусы оптимизаций: деактивирует истекшие и автоматически добавляет новые проблемы.
        
        Со временем в системе должны появляться новые проблемы, а старые оптимизации терять эффективность.
        Этот метод обеспечивает "эволюцию" системы, делая ее более реалистичной.
        
        Returns:
            Словарь с информацией о выполненных действиях.
        """
        try:
            current_time = datetime.now(UTC)
            actions_taken = {
                "expired_optimizations": 0,
                "new_problems_added": 0,
                "automatic_optimizations": 0
            }
            expired_query = (
                select(OptimizationAction)
                .filter(
                    OptimizationAction.is_active == True,
                    OptimizationAction.effective_until < current_time
                )
            )
            expired_result = await self.db.scalars(expired_query)
            expired_optimizations = expired_result.all()
            
            for opt in expired_optimizations:
                opt.is_active = False
                actions_taken["expired_optimizations"] += 1
                logger.info(f"Деактивирована истекшая оптимизация: {opt.slug} - {opt.action_type}")
            traffic_data = await self._fetch_traffic_data(60)  
            if not traffic_data:
                await self.db.flush()
                await self.db.commit()
                return actions_taken
                
            network_status = await self._calculate_network_status(traffic_data)
            status_level = network_status.get("status", "normal")
            problem_probability = {
                "critical": 0.25,  
                "warning": 0.15,   
                "normal": 0.05     
            }.get(status_level, 0.1)
            node_groups = await self._analyze_nodes(traffic_data)
            for nodes in node_groups.values():
                if not nodes or np.random.random() > problem_probability:
                    continue
                problem_node = np.random.choice(nodes)
                problem_types = [
                    "Повышенная нагрузка",
                    "Увеличение задержки",
                    "Потери пакетов",
                    "Нестабильность соединения",
                    "Интерференция сигнала",
                    "Ограниченная пропускная способность",
                    "Несбалансированный трафик"
                ]
                
                problem = np.random.choice(problem_types)
                logger.info(f"Обнаружена новая проблема: {problem} на узле {problem_node.node_id}")
                actions_taken["new_problems_added"] += 1
                if settings.AUTO_OPTIMIZATION and np.random.random() < 0.3:  
                    action_type = f"Автоматическое решение: {problem}"
                    action_type_slug = re.sub(r'[^a-zA-Z0-9-]', '', action_type.lower())
                    slug = f"{action_type_slug}-{uuid.uuid4().hex[:8]}"
                    effective_days = np.random.randint(1, 5)
                    effective_until = current_time + timedelta(days=effective_days)
                    auto_opt = OptimizationAction(
                        slug=slug,
                        action_type=action_type,
                        affected_nodes=[problem_node.node_id],
                        description=f"Автоматическая оптимизация для решения проблемы: {problem}",
                        before_metrics={
                            "bandwidth": problem_node.bandwidth,
                            "latency": problem_node.latency,
                            "packet_loss": problem_node.packet_loss
                        },
                        success=True,
                        created_at=current_time,
                        effective_until=effective_until,
                        is_active=True
                    )
                    self.db.add(auto_opt)
                    actions_taken["automatic_optimizations"] += 1
                    logger.info(f"Применена автоматическая оптимизация: {slug} для {problem_node.node_id}")
            await self.db.flush()
            await self.db.commit()
            return actions_taken
        except Exception as e:
            logger.error(f"Ошибка при обновлении статусов оптимизаций: {e}")
            await self.db.rollback()
            return {"error": str(e)}

    async def get_optimization_status(self) -> dict:
        """Получает текущий статус оптимизации сети.
        
        Returns:
            Словарь с информацией о статусе оптимизации сети.
        """
        try:
            current_time = datetime.now(UTC)
            
            # Получаем активные оптимизации
            active_optimizations = await self.get_active_optimizations()
            
            # Получаем недавние данные о трафике
            traffic_data = await self._fetch_traffic_data(30)
            if not traffic_data:
                return {
                    "status": "unknown",
                    "timestamp": current_time.isoformat(),
                    "message": "Недостаточно данных для определения статуса",
                    "active_optimizations_count": len(active_optimizations),
                    "active_optimizations": active_optimizations[:5] if len(active_optimizations) > 5 else active_optimizations
                }
            
            # Рассчитываем текущий статус сети
            network_status = await self._calculate_network_status(traffic_data)
            
            # Группируем активные оптимизации по типам
            optimization_types = {}
            for opt in active_optimizations:
                opt_type = opt["action_type"]
                if opt_type not in optimization_types:
                    optimization_types[opt_type] = 0
                optimization_types[opt_type] += 1
            
            # Анализируем данные по узлам
            node_groups = await self._analyze_nodes(traffic_data)
            nodes_status = {}
            total_utilization = 0
            total_latency = 0
            total_packet_loss = 0
            node_count = 0
            
            for node_type, nodes in node_groups.items():
                if not nodes:
                    continue
                    
                node_count += len(nodes)
                nodes_status[node_type] = {
                    "count": len(nodes),
                    "avg_utilization": sum(n.bandwidth / n.capacity_mbps for n in nodes) / len(nodes),
                    "avg_latency": sum(n.latency for n in nodes) / len(nodes),
                    "avg_packet_loss": sum(n.packet_loss for n in nodes) / len(nodes),
                    "overloaded_nodes": sum(1 for n in nodes if n.bandwidth / n.capacity_mbps > self.utilization_threshold_high),
                    "underloaded_nodes": sum(1 for n in nodes if n.bandwidth / n.capacity_mbps < self.utilization_threshold_low)
                }
                
                total_utilization += sum(n.bandwidth / n.capacity_mbps for n in nodes)
                total_latency += sum(n.latency for n in nodes)
                total_packet_loss += sum(n.packet_loss for n in nodes)
            
            if node_count > 0:
                avg_utilization = total_utilization / node_count
                avg_latency = total_latency / node_count
                avg_packet_loss = total_packet_loss / node_count
            else:
                avg_utilization = 0
                avg_latency = 0
                avg_packet_loss = 0
            
            # Определяем требуется ли оптимизация
            needs_optimization = (
                network_status["status"] == "critical" or
                avg_utilization > self.utilization_threshold_high or
                any(ns["overloaded_nodes"] > 0 for ns in nodes_status.values())
            )
            
            # Эффективность оптимизации
            optimization_effectiveness = "высокая"
            if network_status["status"] == "critical":
                optimization_effectiveness = "низкая"
            elif network_status["status"] == "warning":
                optimization_effectiveness = "средняя"

            status_message = network_status.get("reason", "Статус сети определен")

            return {
                "status": network_status["status"],
                "timestamp": current_time.isoformat(),
                "message": status_message,
                "network_metrics": {
                    "avg_utilization": round(avg_utilization * 100, 2),
                    "avg_latency": round(avg_latency, 2),
                    "avg_packet_loss": round(avg_packet_loss, 2)
                },
                "nodes_status": nodes_status,
                "active_optimizations_count": len(active_optimizations),
                "optimization_types": optimization_types,
                "top_optimizations": active_optimizations[:3] if active_optimizations else [],
                "needs_optimization": needs_optimization,
                "optimization_effectiveness": optimization_effectiveness,
                "auto_optimization_enabled": settings.AUTO_OPTIMIZATION
            }
        except Exception as e:
            logger.exception(f"Ошибка при получении статуса оптимизации: {e}")
            return {
                "status": "error",
                "message": f"Ошибка при получении статуса оптимизации: {str(e)}",
                "timestamp": datetime.now(UTC).isoformat()
            }

