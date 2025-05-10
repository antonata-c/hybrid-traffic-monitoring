from pydantic import BaseModel
from datetime import datetime


class TrafficData(BaseModel):
    node_id: str
    node_type: str
    bandwidth: float
    latency: float
    packet_loss: float
    timestamp: datetime

    class Config:
        from_attributes = True


class OptimizationResponse(BaseModel):
    message: str
    improvement: float


class TrafficAnalytics(BaseModel):
    average_bandwidth: float
    average_latency: float
    average_packet_loss: float
    high_latency_nodes: list[dict]