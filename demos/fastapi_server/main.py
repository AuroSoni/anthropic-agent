from fastapi import FastAPI

app = FastAPI(
    title="Anthropic Agent API",
    description="Demo API for anthropic-agent package",
    version="0.1.0",
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
