import pandas as pd
import numpy as np
import logging
from datetime import UTC, datetime, timedelta
from typing import List
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from models import Traffic

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Маппинг узлов на маршрутизаторы Abilene
NODE_TO_ROUTER = {node: f"router_{i % 12}" for i, node in enumerate(settings.NETWORK_NODES)}

# Параметры узлов
NODE_TYPES = {
    "fiber": {"base_bandwidth": 1000, "base_latency": 5, "base_packet_loss": 0.1, "capacity": 1000},
    "satellite": {"base_bandwidth": 50, "base_latency": 600, "base_packet_loss": 2, "capacity": 100},
    "5G": {"base_bandwidth": 500, "base_latency": 20, "base_packet_loss": 1, "capacity": 500}
}

async def collect_traffic_data(db: AsyncSession, data_index: int = 0):
    """Собирает данные о трафике, основанные на датасете Abilene и реалистичном моделировании."""
    try:
        abilene_data = pd.read_csv("../data/abilene_traffic.csv")
        logger.info(f"Loaded Abilene data with {len(abilene_data)} rows")
    except FileNotFoundError:
        logger.warning("Abilene data not found, using fallback")
        abilene_data = pd.DataFrame(
            {
                "timestamp": [datetime.now(UTC) - timedelta(minutes=i) for i in range(1000)],
                "router_id": [f"router_{i % 12}" for i in range(1000)],
                "traffic_mbps": [10.0 * (i % 100 + 1) for i in range(1000)]
            }
        )
    try:
        is_peak_hour = 18 <= datetime.now(UTC).hour <= 22
        data_index = data_index % len(abilene_data)  # Циклический доступ к данным
        current_time = datetime.now(UTC)

        for node in settings.NETWORK_NODES:
            node_type = "fiber" if "fiber" in node.lower() else "satellite" if "sat" in node.lower() else "5G"
            params = NODE_TYPES[node_type]

            if node_type == "fiber":
                # Данные из Abilene для fiber
                router_data = abilene_data[abilene_data["router_id"] == NODE_TO_ROUTER[node]]
                if not router_data.empty:
                    row = router_data.iloc[data_index % len(router_data)]
                    bandwidth_mbps = float(row["traffic_mbps"])
                    # Добавляем случайные вариации (±10%) для реалистичности
                    variation = np.random.uniform(0.9, 1.1)
                    bandwidth_mbps *= variation
                    # Рассчитываем загрузку
                    utilization = min(bandwidth_mbps / params["capacity"], 1.0)
                    # Увеличиваем задержку и потери при высокой загрузке
                    latency = params["base_latency"] * (1 + 2 * utilization)  # Более сильное влияние загрузки
                    packet_loss = params["base_packet_loss"] * (1 + 3 * utilization)  # Увеличенные потери
                    if is_peak_hour:
                        bandwidth_mbps *= 1.2  # Увеличение трафика в пиковые часы
                        latency *= 1.5
                        packet_loss *= 1.5
                else:
                    logger.warning(f"No data for router {NODE_TO_ROUTER[node]}")
                    bandwidth_mbps = params["base_bandwidth"] * 0.5
                    latency = params["base_latency"]
                    packet_loss = params["base_packet_loss"]
            elif node_type == "satellite":
                # Моделирование для спутников с учетом помех
                bandwidth_mbps = params["base_bandwidth"] * np.random.uniform(0.7, 1.0)  # Вариации 70-100%
                latency = params["base_latency"] * np.random.uniform(1.0, 1.3)  # Задержка 600-780 мс
                packet_loss = params["base_packet_loss"] * np.random.uniform(1.0, 2.0)  # Потери 2-4%
                if is_peak_hour:
                    bandwidth_mbps *= 0.8  # Снижение пропускной способности
                    latency *= 1.2
                    packet_loss *= 1.3
                # Погодные помехи каждые 15 минут
                if current_time.minute % 15 == 0:
                    latency += np.random.uniform(50, 100)  # Доп. задержка 50-100 мс
                    packet_loss = min(10, packet_loss * 1.5)
            else:  # 5G
                # Моделирование для 5G с учетом перегрузок
                bandwidth_mbps = params["base_bandwidth"] * np.random.uniform(0.8, 1.2)  # Вариации 80-120%
                latency = params["base_latency"] * np.random.uniform(1.0, 1.5)  # Задержка 20-30 мс
                packet_loss = params["base_packet_loss"] * np.random.uniform(1.0, 1.8)  # Потери 1-1.8%
                if is_peak_hour:
                    bandwidth_mbps *= 0.9  # Снижение пропускной способности
                    latency *= 1.3
                    packet_loss *= 1.4
                # Перегрузка сети каждые 10 минут
                if current_time.minute % 10 == 0:
                    bandwidth_mbps *= 0.7
                    latency += np.random.uniform(10, 20)  # Доп. задержка 10-20 мс
                    packet_loss = min(5, packet_loss * 1.2)

            traffic = Traffic(
                node_id=node,
                node_type=node_type,
                bandwidth=round(bandwidth_mbps, 2),
                latency=round(latency, 2),
                packet_loss=round(packet_loss, 2),
                timestamp=current_time
            )
            db.add(traffic)
            logger.info(f"Added traffic data for {node}: bandwidth={bandwidth_mbps:.2f}, latency={latency:.2f}, packet_loss={packet_loss:.2f}")

        await db.commit()
    except Exception as e:
        logger.error(f"Error collecting traffic data: {e}")
        raise

async def get_traffic_data(db: AsyncSession, node_type: str = None) -> List[Traffic]:
    """Возвращает последние данные о трафике, с опциональной фильтрацией по типу узла."""
    query = select(Traffic).order_by(Traffic.timestamp.desc()).limit(100)
    if node_type:
        query = query.filter(Traffic.node_type == node_type)
    result = await db.scalars(query)
    return result.all()

async def get_traffic_analytics(db: AsyncSession):
    """Возвращает аналитику по трафику: средние метрики, тренды и рекомендации."""
    query = select(Traffic).filter(Traffic.timestamp >= datetime.now(UTC) - timedelta(minutes=60))
    result = await db.scalars(query)
    traffic_data = result.all()

    if not traffic_data:
        return {"message": "No data available"}

    avg_bandwidth = sum(t.bandwidth for t in traffic_data) / len(traffic_data)
    avg_latency = sum(t.latency for t in traffic_data) / len(traffic_data)
    avg_packet_loss = sum(t.packet_loss for t in traffic_data) / len(traffic_data)

    high_latency_nodes = [
        {"node_id": t.node_id, "node_type": t.node_type, "latency": t.latency}
        for t in traffic_data if t.latency > (700 if t.node_type == "satellite" else 50)
    ]
    high_loss_nodes = [
        {"node_id": t.node_id, "node_type": t.node_type, "packet_loss": t.packet_loss}
        for t in traffic_data if t.packet_loss > (5 if t.node_type == "satellite" else 2)
    ]

    recent_data = [t for t in traffic_data if t.timestamp >= datetime.now(UTC) - timedelta(minutes=10)]
    latency_trend = "insufficient data"
    if len(recent_data) > 1:
        old_latency = sum(t.latency for t in recent_data[-10:]) / min(len(recent_data), 10)
        new_latency = sum(t.latency for t in recent_data[:10]) / min(len(recent_data), 10)
        latency_trend = "rising" if new_latency > old_latency else "falling" if new_latency < old_latency else "stable"

    recommendations = []
    if high_latency_nodes:
        recommendations.append("Рассмотрите перераспределение трафика с узлов с высокой задержкой.")
    if high_loss_nodes:
        recommendations.append("Проверьте конфигурацию узлов с высоким уровнем потерь пакетов.")

    return {
        "average_bandwidth": round(avg_bandwidth, 2),
        "average_latency": round(avg_latency, 2),
        "average_packet_loss": round(avg_packet_loss, 2),
        "high_latency_nodes": high_latency_nodes,
        "high_loss_nodes": high_loss_nodes,
        "latency_trend": latency_trend,
        "recommendations": recommendations
    }