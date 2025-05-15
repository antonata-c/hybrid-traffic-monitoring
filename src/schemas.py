from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


class TrafficData(BaseModel):
    """Схема для данных о трафике в гибридной сети."""
    node_id: str
    node_type: str
    bandwidth: float
    capacity_mbps: float
    latency: float
    packet_loss: float
    switched_from: Optional[str] = None
    switch_reason: Optional[str] = None
    switch_time: float
    switch_packet_loss: float
    timestamp: datetime

    class Config:
        from_attributes = True


class OptimizationAction(BaseModel):
    """Схема действия по оптимизации трафика."""
    action: str
    target_nodes: list[str]
    description: str
    impact_level: str
    estimated_improvement: float
    implementation_steps: list[str]


class OptimizationResponse(BaseModel):
    """Схема для ответа с рекомендациями по оптимизации."""
    message: str
    recommendations: List[Dict[str, Any]]
    network_status: Dict[str, Any] = Field(default_factory=dict)


class SwitchRoutesRequest(BaseModel):
    """Схема для запроса на переключение маршрутов между узлами."""
    from_node: str
    to_node: str
    traffic_percentage: float = Field(ge=0.0, le=100.0)
    duration_minutes: Optional[int] = Field(default=None, ge=0)
    priority_traffic_only: bool = False


class QoSConfigRequest(BaseModel):
    """Схема для настройки QoS-политик."""
    node_id: str
    traffic_type: str
    priority: int = Field(ge=1, le=5)
    bandwidth_limit: Optional[float] = None
    latency_requirement_ms: Optional[float] = None
    packet_loss_threshold: Optional[float] = None


class NetworkNode(BaseModel):
    """Схема узла сети для отображения топологии."""
    id: str
    type: str
    connections: List[str]
    status: str
    metrics: Dict[str, float]
    coordinates: Dict[str, float] = Field(default_factory=dict)


class NetworkTopology(BaseModel):
    """Схема топологии сети для визуализации."""
    nodes: List[NetworkNode]
    links: List[Dict[str, Any]]
    timestamp: datetime
