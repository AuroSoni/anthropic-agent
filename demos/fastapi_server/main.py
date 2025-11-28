"""FastAPI server with health check and database integration."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from db import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: nothing needed (lazy pool initialization)
    yield
    # Shutdown: close database connection pool
    await db.close()


app = FastAPI(
    title="Anthropic Agent API",
    description="Demo API for anthropic-agent package",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint with database connectivity status.
    
    Returns:
        Health status including database connectivity check.
    """
    db_health = await db.health_check()
    
    overall_status = "healthy"
    if db_health.get("status") == "down":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "services": {
            "database": db_health,
        },
    }
