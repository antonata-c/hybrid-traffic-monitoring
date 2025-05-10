from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db
from schemas import TrafficAnalytics, TrafficData
from services.traffic import collect_traffic_data, get_traffic_analytics, get_traffic_data

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/traffic", response_model=list[TrafficData])
async def get_traffic(node_type: str | None = None, db: AsyncSession = Depends(get_db)):
    return await get_traffic_data(db, node_type)

@router.get("/analytics", response_model=TrafficAnalytics)
async def analytics(db: AsyncSession = Depends(get_db)):
    return await get_traffic_analytics(db)
