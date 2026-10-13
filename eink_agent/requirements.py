"""Validated device requirements extracted from user language."""

from typing import Literal

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
    # A free-form string let the model send values such as "阅读器", which
    # silently matched nothing; unknown values now fail validation and are
    # reported back to the model.
    category: Literal["reader", "note", "tablet"] | None = Field(
        default=None,
        description=(
            "reader = e-book reader, note = handwriting notebook, "
            "tablet = general-purpose E Ink tablet. Leave empty when the user "
            "does not clearly ask for one type."
        ),
    )
    brand: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Exact manufacturer name, e.g. Amazon (for Kindle), BOOX, iReader, "
            "Kobo. Use keyword for product or series names."
        ),
    )
    keyword: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Words that must all appear in the brand or model name, "
            "e.g. 'Kindle Paperwhite' or 'Clara BW'."
        ),
    )
