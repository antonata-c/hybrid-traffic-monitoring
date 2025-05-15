from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

from enums import NodeType, TrafficType

Base = declarative_base()


class Traffic(Base):
    """Модель для хранения данных о трафике в гибридной сети."""

    __tablename__ = "traffic"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    node_type: Mapped[str] = mapped_column(String, index=True)
    bandwidth: Mapped[float]
    latency: Mapped[float]
    packet_loss: Mapped[float]
    capacity_mbps: Mapped[float]
    switched_from: Mapped[str | None] = mapped_column(String, nullable=True)
    switch_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    switch_time: Mapped[float] = mapped_column(Float, default=0.0)
    switch_packet_loss: Mapped[float] = mapped_column(Float, default=0.0)
    jitter: Mapped[float] = mapped_column(Float, default=0.0)
    signal_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    interference_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Node(Base):
    """Модель для хранения информации об узлах сети."""

    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String, unique=True, index=True)
    node_type: Mapped[str] = mapped_column(Enum(NodeType), index=True)
    description: Mapped[str] = mapped_column(String, nullable=True)
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    coordinates: Mapped[dict] = mapped_column(JSON, nullable=True)
    max_capacity: Mapped[float] = mapped_column(Float)
    backup_node: Mapped[str | None] = mapped_column(String, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=3)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now)


class QoSPolicy(Base):
    """Модель для хранения QoS-политик для различных типов трафика."""

    __tablename__ = "qos_policies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    traffic_type: Mapped[str] = mapped_column(Enum(TrafficType), index=True)
    priority: Mapped[int] = mapped_column(Integer)
    bandwidth_reserved: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_latency: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_packet_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)


class OptimizationAction(Base):
    """Модель для хранения истории оптимизационных действий."""

    __tablename__ = "optimization_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String, unique=True, index=True)
    action_type: Mapped[str] = mapped_column(String, index=True)
    affected_nodes: Mapped[list] = mapped_column(JSON)
    description: Mapped[str] = mapped_column(String)
    before_metrics: Mapped[dict] = mapped_column(JSON)
    after_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    improvement: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)
    effective_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class NetworkLink(Base):
    """Модель для хранения связей между узлами сети."""

    __tablename__ = "network_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_node: Mapped[str] = mapped_column(String, index=True)
    target_node: Mapped[str] = mapped_column(String, index=True)
    bandwidth: Mapped[float] = mapped_column(Float)
    latency: Mapped[float] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    link_type: Mapped[str] = mapped_column(String)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.now, onupdate=datetime.now)
