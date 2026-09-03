"""Validated device requirements extracted from user language."""

from pydantic import BaseModel, ConfigDict, Field

class DeviceRequirements(BaseModel):
    """Structured search constraints for E Ink devices."""

    model_config = ConfigDict(extra="forbid")

    max_price: int | None = Field(default=None, ge=0)
    min_price: int | None = Field(default=None, ge=0)
    min_screen_size: float | None = Field(default=None, gt=0)
    max_screen_size: float | None = Field(default=None, gt=0)
    max_weight_g: int | None = Field(default=None, gt=0)
    supports_stylus: bool | None = None
    is_color: bool | None = None
    is_open_system: bool | None = None
    category: str | None = Field(default=None, min_length=1)
    brand: str | None = Field(default=None, min_length=1)
