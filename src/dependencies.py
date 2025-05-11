from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=7,
    max_overflow=20,
    pool_pre_ping=True,
)

session_factory = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)


async def get_db():
    """Предоставляет асинхронную сессию базы данных."""
    db = session_factory()
    try:
        yield db
    finally:
        await db.close()
