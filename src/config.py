from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Конфигурация приложения с загрузкой переменных из .env."""
    DATABASE_URL: str
    NETWORK_NODES: list[str]
    TOTAL_TRAFFIC_DEMAND: float
    DATA_SOURCE: str  
    SNMP_HOST: str
    SNMP_PORT: int
    SNMP_COMMUNITY: str
    ANALYSIS_WINDOW_MINUTES: int = 60
    FORECAST_WINDOW_MINUTES: int = 30
    ANOMALY_DETECTION_SENSITIVITY: float = 1.5
    LATENCY_THRESHOLD_FIBER: float = 15.0  
    LATENCY_THRESHOLD_5G: float = 30.0  
    LATENCY_THRESHOLD_SATELLITE: float = 500.0  
    PACKET_LOSS_THRESHOLD: float = 2.0  
    HIGH_UTILIZATION_THRESHOLD: float = 80.0  
    LOW_UTILIZATION_THRESHOLD: float = 20.0  
    OPTIMIZATION_INTERVAL_MINUTES: int = 15
    ROUTE_STABILITY_TIME_SECONDS: int = 300
    QOS_ENABLED: bool = True
    ENABLE_SDN_CONTROL: bool = False
    AUTO_OPTIMIZATION: bool = False
    LOAD_BALANCER_ALGORITHM: str = "weighted_round_robin"  
    TRAFFIC_SHAPING_ENABLED: bool = True
    CONGESTION_CONTROL_ALGORITHM: str = "aimd"  


settings = Settings()
