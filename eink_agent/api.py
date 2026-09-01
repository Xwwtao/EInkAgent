"""FastAPI application for EInkAgent."""
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query

from device_repository import search_devices, get_device_detail


app = FastAPI(title="EInkAgent", version="0.2.0")

@app.get("/health")
def health() -> dict[str,str]:
    """Return the service health status"""
    return {"status": "ok"}


@app.get("/devices")
def get_devices(
    brand: str | None = None,
    max_price: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict[str, Any]]:
    """Search devices using HTTP query parameters."""
    return search_devices(
        brand=brand,
        max_price=max_price,
        limit=limit,
    )


@app.get("/devices/{device_id}")
def get_device(
    device_id: Annotated[int, Path(ge=1)],
) -> dict[str, Any]:
    """Return one device by its ID."""
    device = get_device_detail(device_id)

    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")

    return device
