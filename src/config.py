from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/traffic_db"
    NETWORK_NODES: list = [
        "fiber_node1", "fiber_node2", "sat_node1", "sat_node2", "5g_node1", "5g_node2"
    ]

settings = Settings()
