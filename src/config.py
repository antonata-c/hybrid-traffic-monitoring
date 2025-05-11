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


settings = Settings()
