"""FastAPI application for EInkAgent."""

from fastapi import FastAPI

app = FastAPI(title="EInkAgent", version="0.2.0")

@app.get("/health")
def health() -> dict[str,str]:
    """Return the service health status"""
    return {"status": "ok"}