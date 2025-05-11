from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()


class Traffic(Base):
    """Модель для хранения данных о трафике."""
    __tablename__ = "traffic"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    node_type: Mapped[str] = mapped_column(String, index=True)
    bandwidth: Mapped[float]
    latency: Mapped[float]
    packet_loss: Mapped[float]
    capacity_mbps: Mapped[float]
    switched_from: Mapped[str] = mapped_column(nullable=True)
    switch_reason: Mapped[str] = mapped_column(nullable=True)
    switch_time: Mapped[float] = mapped_column(default=0.0)
    switch_packet_loss: Mapped[float] = mapped_column(default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
