from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db
from enums import NodeType
from schemas import OptimizationResponse
from services.analyzer import TrafficAnalyzer
from services.optimizer import TrafficOptimizer

router = APIRouter()


@router.get("/analytics")
async def get_analytics(node_type: NodeType | None = None, minutes: int = 60, db: AsyncSession = Depends(get_db)):
    analyzer = TrafficAnalyzer(db)
    return await analyzer.get_traffic_analytics(node_type, minutes)


@router.get("/analytics/{node_id}")
async def get_node_analytics(node_id: str, minutes: int = 60, db: AsyncSession = Depends(get_db)):
    analyzer = TrafficAnalyzer(db)
    result = await analyzer.get_detailed_node_analytics(node_id, minutes)
    if "message" in result:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@router.get("/optimize", response_model=OptimizationResponse)
async def optimize(db: AsyncSession = Depends(get_db)):
    """Предоставляет рекомендации по оптимизации трафика."""
    try:
        optimizer = TrafficOptimizer(db)
        return await optimizer.optimize_traffic()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
