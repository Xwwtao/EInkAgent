"""FastAPI application for EInkAgent."""

from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field, PositiveInt

from device_repository import compare_devices, get_device_detail, search_devices


class DeviceComparisonRequest(BaseModel):
    """Validated request body for comparing devices."""

    device_ids: list[PositiveInt] = Field(min_length=2, max_length=5)


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


@app.post("/devices/compare")
def compare_device_options(
    request: DeviceComparisonRequest,
) -> list[dict[str, Any]]:
    """Compare devices supplied in a JSON request body."""
    try:
        return compare_devices(request.device_ids)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/devices/{device_id}")
def get_device(
    device_id: Annotated[int, Path(ge=1)],
) -> dict[str, Any]:
    """Return one device by its ID."""
    device = get_device_detail(device_id)

    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")

    return device
