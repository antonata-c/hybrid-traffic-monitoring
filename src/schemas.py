from datetime import datetime

from pydantic import BaseModel


class TrafficData(BaseModel):
    """Схема для данных о трафике."""
    node_id: str
    node_type: str
    bandwidth: float
    capacity_mbps: float
    latency: float
    packet_loss: float
    switched_from: str | None
    switch_reason: str | None
    switch_time: float
    switch_packet_loss: float
    timestamp: datetime

    class Config:
        from_attributes = True


class OptimizationResponse(BaseModel):
    """Схема для ответа с рекомендациями по оптимизации."""
    message: str
    recommendations: list[dict]
