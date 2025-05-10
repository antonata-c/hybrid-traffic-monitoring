from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()


class Traffic(Base):
    __tablename__ = "traffic"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String, index=True)
    node_type: Mapped[str] = mapped_column(String, index=True)
    bandwidth: Mapped[float]
    latency: Mapped[float]
    packet_loss: Mapped[float]
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
