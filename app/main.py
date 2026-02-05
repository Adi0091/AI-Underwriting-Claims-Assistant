from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.query import router as query_router

app = FastAPI(
    title="AI-Powered Underwriting & Claims Triage Assistant",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(query_router)
