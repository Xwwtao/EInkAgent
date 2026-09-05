"""Tests for structured device requirements."""

import pytest
from pydantic import ValidationError

from device_repository import search_devices
from eink_agent.requirements import DeviceRequirements


def test_requirements_accept_valid_search_constraints():
    requirements = DeviceRequirements.model_validate(
        {
            "max_price": 2000,
            "max_weight_g": 300,
            "supports_stylus": True,
        }
    )

    assert requirements.model_dump(exclude_none=True) == {
        "max_price": 2000,
        "max_weight_g": 300,
        "supports_stylus": True,
    }


def test_requirements_leave_unknown_constraints_unset():
    requirements = DeviceRequirements()

    assert requirements.model_dump(exclude_none=True) == {}


def test_requirements_reject_negative_price():
    with pytest.raises(ValidationError):
        DeviceRequirements(max_price=-1)


def test_requirements_reject_unexpected_fields():
    with pytest.raises(ValidationError):
        DeviceRequirements.model_validate({"battery_days": 30})


def test_requirements_can_drive_device_search():
    requirements = DeviceRequirements(
        max_price=2000,
        max_weight_g=300,
        supports_stylus=True,
    )

    devices = search_devices(
        **requirements.model_dump(exclude_none=True)
    )

    assert [device["model"] for device in devices] == ["Color 7"]
