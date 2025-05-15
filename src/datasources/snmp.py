import logging
from datetime import UTC, datetime
from typing import Dict, List, Optional, Any

import numpy as np
from pysnmp.hlapi.v3arch.asyncio import *
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from enums import NodeType, TrafficType
from models import Node, OptimizationAction
from .base import DataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SNMPDataSource(DataSource):
    """Источник данных, использующий SNMP для сбора метрик из реальной сети с поддержкой всех типов узлов."""

    def __init__(self, db: Optional[AsyncSession] = None):
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
                "oid_in": "1.3.6.1.2.1.2.2.1.10",
                "oid_out": "1.3.6.1.2.1.2.2.1.16",
                "oid_errors": "1.3.6.1.2.1.2.2.1.14"
            },
            NodeType.gen5: {
                "base_bandwidth": 500, 
                "base_latency": 20, 
                "base_packet_loss": 1, 
                "capacity": 500,
                "jitter": 8,
                "switch_time": 0.2,
                "switch_loss": 1.0,
                "oid_in": "1.3.6.1.2.1.2.2.1.10", 
                "oid_out": "1.3.6.1.2.1.2.2.1.16",
                "oid_errors": "1.3.6.1.2.1.2.2.1.14"
            },
            NodeType.satellite: {
                "base_bandwidth": 50, 
                "base_latency": 600, 
                "base_packet_loss": 2, 
                "capacity": 100,
                "jitter": 50,
                "switch_time": 0.5,
                "switch_loss": 2.0,
                "oid_in": "1.3.6.1.2.1.2.2.1.10", 
                "oid_out": "1.3.6.1.2.1.2.2.1.16",
                "oid_errors": "1.3.6.1.2.1.2.2.1.14"
            },
            NodeType.microwave: {
                "base_bandwidth": 200, 
                "base_latency": 30, 
                "base_packet_loss": 1.5, 
                "capacity": 300,
                "jitter": 10,
                "switch_time": 0.3,
                "switch_loss": 1.2,
                "oid_in": "1.3.6.1.2.1.2.2.1.10", 
                "oid_out": "1.3.6.1.2.1.2.2.1.16",
                "oid_errors": "1.3.6.1.2.1.2.2.1.14"
            },
            NodeType.starlink: {
                "base_bandwidth": 150, 
                "base_latency": 80, 
                "base_packet_loss": 1.8, 
                "capacity": 250,
                "jitter": 15,
                "switch_time": 0.4,
                "switch_loss": 1.5,
                "oid_in": "1.3.6.1.2.1.2.2.1.10", 
                "oid_out": "1.3.6.1.2.1.2.2.1.16",
                "oid_errors": "1.3.6.1.2.1.2.2.1.14"
            },
            NodeType.hybrid: {
                "base_bandwidth": 600, 
                "base_latency": 40, 
                "base_packet_loss": 1.2, 
                "capacity": 800,
                "jitter": 12,
                "switch_time": 0.25,
                "switch_loss": 1.1,
                "oid_in": "1.3.6.1.2.1.2.2.1.10", 
                "oid_out": "1.3.6.1.2.1.2.2.1.16",
                "oid_errors": "1.3.6.1.2.1.2.2.1.14"
            }
        }
        
        self.traffic_types = {
            TrafficType.VOICE: {
                "bandwidth_factor": 0.05,
                "latency_sensitivity": 2.0,
                "jitter_sensitivity": 2.0,
                "packet_loss_sensitivity": 2.0
            },
            TrafficType.VIDEO: {
                "bandwidth_factor": 0.4,
                "latency_sensitivity": 1.5,
                "jitter_sensitivity": 1.8,
                "packet_loss_sensitivity": 1.7
            },
            TrafficType.INTERACTIVE: {
                "bandwidth_factor": 0.2,
                "latency_sensitivity": 1.8,
                "jitter_sensitivity": 1.6,
                "packet_loss_sensitivity": 1.4
            },
            TrafficType.STREAMING: {
                "bandwidth_factor": 0.3,
                "latency_sensitivity": 1.2,
                "jitter_sensitivity": 1.5,
                "packet_loss_sensitivity": 1.3
            },
            TrafficType.DATA: {
                "bandwidth_factor": 0.15,
                "latency_sensitivity": 1.0,
                "jitter_sensitivity": 1.0,
                "packet_loss_sensitivity": 1.2
            },
            TrafficType.IOT: {
                "bandwidth_factor": 0.05,
                "latency_sensitivity": 1.1,
                "jitter_sensitivity": 1.1,
                "packet_loss_sensitivity": 1.2
            },
            TrafficType.SIGNALING: {
                "bandwidth_factor": 0.02,
                "latency_sensitivity": 1.7,
                "jitter_sensitivity": 1.3,
                "packet_loss_sensitivity": 1.8
            }
        }
        
        self.previous_counters = {}
        self.last_timestamp = None
        self.applied_optimizations = {}
        self.last_optimizations_check = None

    async def _get_nodes_from_db(self) -> List[Dict]:
        """Получает список узлов из базы данных."""
        if not self.db:
            logger.info("БД недоступна, использую NETWORK_NODES из настроек")
            network_nodes = settings.NETWORK_NODES
            nodes = []
            for node_id in network_nodes:
                node_type = self._detect_node_type(node_id)
                params = self.node_types[node_type]
                nodes.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "max_capacity": params["capacity"]
                })
            logger.info(f"Созданы узлы из настроек: {nodes}")
            return nodes
            
        query = select(Node).filter(Node.is_active == True)
        result = await self.db.scalars(query)
        nodes = result.all()
        
        if not nodes:
            logger.warning("В БД не найдены узлы, использую NETWORK_NODES из настроек")
            network_nodes = settings.NETWORK_NODES
            nodes = []
            for node_id in network_nodes:
                node_type = self._detect_node_type(node_id)
                params = self.node_types[node_type]
                nodes.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "max_capacity": params["capacity"]
                })
            logger.info(f"Созданы узлы из настроек: {nodes}")
            return nodes
            
        nodes_data = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "max_capacity": node.max_capacity
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
            "hybrid": NodeType.hybrid
        }
        
        for key, value in node_types.items():
            if key in node_id.lower():
                return value
                
        return NodeType.gen5

    async def _get_snmp_value(self, node_id: str, oid: str, interface_index: int = 1) -> Optional[int]:
        """Запрашивает значение по SNMP для указанного OID и интерфейса.
        
        Args:
            node_id: Идентификатор узла
            oid: OID для запроса
            interface_index: Индекс интерфейса
            
        Returns:
            Значение счетчика или None в случае ошибки
        """
        try:
            ip_address = f"{settings.SNMP_HOST}"
            full_oid = f"{oid}.{interface_index}"
            
            error_indication, error_status, error_index, var_binds = await get_cmd(
                SnmpEngine(),
                CommunityData(settings.SNMP_COMMUNITY),
                await UdpTransportTarget.create((ip_address, settings.SNMP_PORT)),
                ContextData(),
                ObjectType(ObjectIdentity(full_oid))
            )

            if error_indication:
                logger.error(f"SNMP ошибка для {node_id}: {error_indication}")
                return None
            elif error_status:
                logger.error(f"SNMP статус ошибки для {node_id}: {error_status}")
                return None
            else:
                return int(var_binds[0][1])
        except Exception as e:
            logger.error(f"Ошибка при запросе SNMP для {node_id} ({oid}): {e}")
            return None
    
    async def _calculate_rate(self, node_id: str, counter_name: str, current_value: int, timestamp: datetime) -> float:
        """Рассчитывает скорость изменения счетчика в Mbps.
        
        Args:
            node_id: Идентификатор узла
            counter_name: Имя счетчика
            current_value: Текущее значение счетчика
            timestamp: Временная метка измерения
            
        Returns:
            Скорость в Mbps
        """
        counter_key = f"{node_id}_{counter_name}"
        
        if counter_key not in self.previous_counters or self.last_timestamp is None:
            self.previous_counters[counter_key] = current_value
            self.last_timestamp = timestamp
            return 0.0
            
        value_diff = current_value - self.previous_counters[counter_key]
        
        if value_diff < 0:
            value_diff += 2**32
            
        time_diff = (timestamp - self.last_timestamp).total_seconds()
        if time_diff <= 0:
            return 0.0
            
        rate_mbps = (value_diff * 8) / (time_diff * 1_000_000)
        
        self.previous_counters[counter_key] = current_value
        
        return rate_mbps

    async def _get_node_metrics(self, node: Dict, timestamp: datetime) -> Dict:
        """Получает метрики для указанного узла через SNMP.
        
        Args:
            node: Информация об узле
            timestamp: Временная метка
            
        Returns:
            Словарь с метриками узла
        """
        node_id = node["node_id"]
        node_type = node["node_type"]
        params = self.node_types[node_type]
        
        in_octets = await self._get_snmp_value(node_id, params["oid_in"])
        out_octets = await self._get_snmp_value(node_id, params["oid_out"])
        errors = await self._get_snmp_value(node_id, params["oid_errors"])
        
        if in_octets is None or out_octets is None:
            bandwidth = params["base_bandwidth"] * np.random.uniform(0.3, 0.7)
            utilization = bandwidth / params["capacity"]
            latency = params["base_latency"] * (1 + utilization)
            packet_loss = params["base_packet_loss"] * (1 + utilization * 2)
            jitter = params["jitter"] * np.random.uniform(0.5, 1.5)
            return self._get_default_metrics(node, bandwidth, latency, packet_loss, jitter, timestamp)
            
        in_rate = await self._calculate_rate(node_id, "in", in_octets, timestamp)
        out_rate = await self._calculate_rate(node_id, "out", out_octets, timestamp)
        
        bandwidth = max(in_rate, out_rate)
        
        if bandwidth == 0:
            bandwidth = params["base_bandwidth"] * np.random.uniform(0.3, 0.7)
            
        utilization = bandwidth / params["capacity"]
        latency = params["base_latency"] * (1 + utilization * 2)
        
        if errors is not None:
            error_rate = await self._calculate_rate(node_id, "errors", errors, timestamp)
            packet_loss = (error_rate / (in_rate + 0.001)) * 100
        else:
            packet_loss = params["base_packet_loss"] * (1 + utilization * 3)
            
        jitter = params["jitter"] * (1 + utilization * 1.5)
        
        self.last_timestamp = timestamp
        
        return self._get_default_metrics(node, bandwidth, latency, packet_loss, jitter, timestamp)
    
    def _get_default_metrics(self, node: Dict, bandwidth: float, latency: float, packet_loss: float, jitter: float, timestamp: datetime) -> Dict:
        """Формирует словарь со всеми необходимыми метриками для узла.
        
        Args:
            node: Информация об узле
            bandwidth: Пропускная способность в Mbps
            latency: Задержка в мс
            packet_loss: Потери пакетов в %
            jitter: Джиттер в мс
            timestamp: Временная метка
            
        Returns:
            Словарь с метриками
        """
        node_id = node["node_id"]
        node_type = node["node_type"]
        params = self.node_types[node_type]
        
        signal_strength = None
        interference_level = None
        error_rate = None
        
        if node_type in [NodeType.satellite, NodeType.gen5, NodeType.microwave, NodeType.starlink]:
            signal_strength = -60 - np.random.uniform(0, 40)
            interference_level = np.random.uniform(0, 15)
            error_rate = packet_loss / 100 * np.random.uniform(0.8, 1.5)
            
        return {
            "node_id": node_id,
            "node_type": node_type,
            "bandwidth": bandwidth,
            "latency": latency,
            "packet_loss": packet_loss,
            "capacity_mbps": params["capacity"],
            "jitter": jitter,
            "switched_from": None,
            "switch_reason": None,
            "switch_time": 0.0,
            "switch_packet_loss": 0.0,
            "signal_strength": signal_strength,
            "interference_level": interference_level,
            "error_rate": error_rate,
            "timestamp": timestamp
        }

    async def collect_data(self, nodes: List[str], timestamp: datetime) -> List[Dict]:
        """Собирает данные из сети через SNMP асинхронно."""
        all_nodes = await self._get_nodes_from_db()
        
        target_nodes = [node for node in all_nodes if node["node_id"] in nodes]
        if not target_nodes:
            target_nodes = all_nodes
            
        result = []
        
        for node in target_nodes:
            metrics = await self._get_node_metrics(node, timestamp)
            
            updated_metrics = await self._apply_optimization_effects(metrics["node_id"], metrics["node_type"], metrics)
            
            result.append(updated_metrics)
            
        return result

    async def _get_active_optimizations(self) -> List[Dict]:
        """Получает активные оптимизации из базы данных."""
        if not self.db:
            return []
            
        current_time = datetime.now(UTC)
        
        if (self.last_optimizations_check is None or 
                (current_time - self.last_optimizations_check).total_seconds() > 300):
            
            query = (
                select(OptimizationAction)
                .filter(
                    OptimizationAction.is_active == True,
                    (OptimizationAction.effective_until == None) | (OptimizationAction.effective_until > current_time)
                )
            )
            result = await self.db.scalars(query)
            optimizations = result.all()
            
            self.applied_optimizations = {
                opt.slug: {
                    "action_type": opt.action_type,
                    "affected_nodes": opt.affected_nodes,
                    "created_at": opt.created_at,
                    "effective_until": opt.effective_until
                } for opt in optimizations
            }
            
            self.last_optimizations_check = current_time
            
        return list(self.applied_optimizations.values())

    async def _apply_optimization_effects(self, node_id: str, node_type: str, metrics: Dict) -> Dict:
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
            
            days_since_applied = (datetime.now(UTC) - opt["created_at"]).days
            
            efficacy = max(0.5, 1.0 - days_since_applied * 0.1)
            
            days_until_expiry = (opt["effective_until"] - datetime.now(UTC)).days if opt["effective_until"] else 999
            
            if days_until_expiry < 2:
                efficacy *= 0.5
            
            if "балансировка" in action_type or "load balancing" in action_type:
                reduction_factor = np.random.uniform(0.7, 0.9) * efficacy
                if metrics["bandwidth"] / metrics["capacity_mbps"] > 0.7:
                    updated_metrics["bandwidth"] *= reduction_factor
                    
            elif "qos" in action_type or "приоритизация" in action_type:
                latency_improvement = np.random.uniform(0.8, 0.95) * efficacy
                packet_loss_improvement = np.random.uniform(0.6, 0.85) * efficacy
                updated_metrics["latency"] *= latency_improvement
                updated_metrics["packet_loss"] *= packet_loss_improvement
                
            elif "переключение" in action_type or "switch" in action_type:
                switch_improvement = 0.7 * efficacy
                updated_metrics["switch_time"] *= switch_improvement
                updated_metrics["switch_packet_loss"] *= switch_improvement
                
            elif "пропускная способность" in action_type or "capacity" in action_type:
                capacity_improvement = np.random.uniform(1.05, 1.15) * efficacy
                updated_metrics["capacity_mbps"] *= capacity_improvement
                
            elif "маршрутизация" in action_type or "routing" in action_type:
                if node_type == NodeType.hybrid:
                    updated_metrics["latency"] *= 0.85 * efficacy
                    
            elif "шейпинг" in action_type or "shaping" in action_type:
                if updated_metrics["jitter"] > 5:
                    updated_metrics["jitter"] *= 0.8 * efficacy
                    
            elif "отказоустойчивость" in action_type or "failover" in action_type:
                updated_metrics["packet_loss"] *= 0.85 * efficacy
        
        return updated_metrics
