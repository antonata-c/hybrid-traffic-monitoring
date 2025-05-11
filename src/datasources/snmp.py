import logging
from datetime import datetime

from pysnmp.hlapi.v3arch.asyncio import *

from config import settings
from .base import DataSource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SNMPDataSource(DataSource):
    """Источник данных, использующий SNMP для сбора метрик из реальной сети."""

    def __init__(self):
        self.node_types = {
            "fiber": {"base_bandwidth": 1000, "base_latency": 5, "base_packet_loss": 0.1, "capacity": 1000},
            "satellite": {"base_bandwidth": 50, "base_latency": 600, "base_packet_loss": 2, "capacity": 100},
            "5G": {"base_bandwidth": 500, "base_latency": 20, "base_packet_loss": 1, "capacity": 500}
        }

    def _get_node_type(self, node: str) -> str:
        """Возвращает тип узла."""
        if "fiber" in node.lower():
            return "fiber"
        elif "sat" in node.lower():
            return "satellite"
        return "5G"

    async def collect_data(self, nodes: list[str], timestamp: datetime) -> list[dict]:
        """Собирает данные из сети через SNMP асинхронно."""
        data = []

        for node in nodes:
            node_type = self._get_node_type(node)
            params = self.node_types[node_type]
            bandwidth_mbps = params["base_bandwidth"] * 0.5  

            try:
                
                error_indication, error_status, error_index, var_binds = await get_cmd(
                    SnmpEngine(),
                    CommunityData(settings.SNMP_COMMUNITY),
                    await UdpTransportTarget.create((settings.SNMP_HOST, settings.SNMP_PORT)),
                    ContextData(),
                    ObjectType(ObjectIdentity('IF-MIB', 'ifInOctets', 1))
                )

                if error_indication:
                    logger.error(f"SNMP error for {node}: {error_indication}")
                elif error_status:
                    logger.error(f"SNMP error status for {node}: {error_status}")
                else:
                    
                    bandwidth_mbps = float(var_binds[0][1]) / 1_000_000  

            except Exception as e:
                logger.error(f"Ошибка SNMP для {node}: {e}")

            utilization = min(bandwidth_mbps / params["capacity"], 1.0)
            latency = params["base_latency"] * (1 + 2 * utilization)
            packet_loss = params["base_packet_loss"] * (1 + 3 * utilization)

            data.append(
                {
                    "node_id": node,
                    "node_type": node_type,
                    "bandwidth": round(bandwidth_mbps, 2),
                    "capacity_mbps": params["capacity"],
                    "latency": round(latency, 2),
                    "packet_loss": round(packet_loss, 2),
                    "switched_from": None,
                    "switch_reason": None,
                    "switch_time": 0.0,
                    "switch_packet_loss": 0.0,
                    "timestamp": timestamp
                }
            )

        return data
