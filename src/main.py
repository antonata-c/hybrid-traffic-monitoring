from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from endpoints import router

app = FastAPI(title="Система мониторинга и оптимизации телекоммуникационного трафика в гибридных сетях связи")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
