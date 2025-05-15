from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TrafficData(BaseModel):
    """Схема для данных о трафике в гибридной сети."""

    node_id: str
    node_type: str
    bandwidth: float
    capacity_mbps: float
    latency: float
    packet_loss: float
    switched_from: str | None = None
    switch_reason: str | None = None
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
    recommendations: list[dict[str, Any]]
    network_status: dict[str, Any] = Field(default_factory=dict)


class SwitchRoutesRequest(BaseModel):
    """Схема для запроса на переключение маршрутов между узлами."""

    from_node: str
    to_node: str
    traffic_percentage: float = Field(ge=0.0, le=100.0)
    duration_minutes: int | None = Field(default=None, ge=0)
    priority_traffic_only: bool = False


class QoSConfigRequest(BaseModel):
    """Схема для настройки QoS-политик."""

    node_id: str
    traffic_type: str
    priority: int = Field(ge=1, le=5)
    bandwidth_limit: float | None = None
    latency_requirement_ms: float | None = None
    packet_loss_threshold: float | None = None


class NetworkNode(BaseModel):
    """Схема узла сети для отображения топологии."""

    id: str
    type: str
    connections: list[str]
    status: str
    metrics: dict[str, float]
    coordinates: dict[str, float] = Field(default_factory=dict)


class NetworkTopology(BaseModel):
    """Схема топологии сети для визуализации."""

    nodes: list[NetworkNode]
    links: list[dict[str, Any]]
    timestamp: datetime
