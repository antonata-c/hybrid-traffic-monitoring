from fastapi import FastAPI

from endpoints import router

app = FastAPI(title="Telecom Traffic Monitoring System")
app.include_router(router)
