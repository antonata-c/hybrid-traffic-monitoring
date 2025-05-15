from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db
from enums import NodeType
from schemas import OptimizationResponse, SwitchRoutesRequest, QoSConfigRequest
from services.analyzer import TrafficAnalyzer
from services.optimizer import TrafficOptimizer

router = APIRouter()


@router.get("/analytics")
async def get_analytics(node_type: NodeType | None = None, minutes: int = 60, db: AsyncSession = Depends(get_db)):
    """Получение общей аналитики по трафику в гибридной сети."""
    analyzer = TrafficAnalyzer(db)
    return await analyzer.get_traffic_analytics(node_type, minutes)


@router.get("/analytics/{node_id}")
async def get_node_analytics(node_id: str, minutes: int = 60, db: AsyncSession = Depends(get_db)):
    """Детальная аналитика по трафику на конкретном узле гибридной сети."""
    analyzer = TrafficAnalyzer(db)
    result = await analyzer.get_detailed_node_analytics(node_id, minutes)
    if "message" in result:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@router.get("/optimize", response_model=OptimizationResponse)
async def optimize(db: AsyncSession = Depends(get_db)):
    """Получение рекомендаций по оптимизации трафика в гибридной сети."""
    try:
        optimizer = TrafficOptimizer(db)
        return await optimizer.optimize_traffic()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/apply", response_model=dict)
async def apply_optimization(recommendation_id: int, db: AsyncSession = Depends(get_db)):
    """Применение выбранной рекомендации по оптимизации трафика."""
    try:
        optimizer = TrafficOptimizer(db)
        recommendations = await optimizer.optimize_traffic()
        if recommendation_id >= len(recommendations["recommendations"]):
            raise HTTPException(status_code=404, detail="Рекомендация не найдена")
        
        result = await optimizer.apply_optimization(recommendations["recommendations"][recommendation_id])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/active", response_model=list)
async def get_active_optimizations(db: AsyncSession = Depends(get_db)):
    """Получение списка активных оптимизаций."""
    try:
        optimizer = TrafficOptimizer(db)
        return await optimizer.get_active_optimizations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/routes", response_model=dict)
async def switch_routes(request: SwitchRoutesRequest, db: AsyncSession = Depends(get_db)):
    """Ручное переключение маршрутов между узлами гибридной сети."""
    try:
        optimizer = TrafficOptimizer(db)
        result = await optimizer.manual_route_switch(
            from_node=request.from_node,
            to_node=request.to_node,
            traffic_percentage=request.traffic_percentage
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/qos", response_model=dict)
async def configure_qos(request: QoSConfigRequest, db: AsyncSession = Depends(get_db)):
    """Настройка параметров QoS для приоритизации трафика."""
    try:
        optimizer = TrafficOptimizer(db)
        result = await optimizer.configure_qos_policies(
            node_id=request.node_id,
            traffic_type=request.traffic_type,
            priority=request.priority
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimize/status")
async def get_optimization_status(db: AsyncSession = Depends(get_db)):
    """Получение текущего статуса оптимизации сети."""
    try:
        optimizer = TrafficOptimizer(db)
        return await optimizer.get_optimization_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/network/topology")
async def get_network_topology(db: AsyncSession = Depends(get_db)):
    """Получение топологии сети с информацией о связях между узлами."""
    try:
        analyzer = TrafficAnalyzer(db)
        return await analyzer.get_network_topology()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/update-statuses", response_model=dict)
async def update_optimization_statuses(db: AsyncSession = Depends(get_db)):
    """Обновляет статусы оптимизаций: деактивирует истекшие и добавляет новые проблемы."""
    try:
        optimizer = TrafficOptimizer(db)
        result = await optimizer.update_optimization_statuses()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
