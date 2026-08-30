"""Behavior tests for the E Ink device repository tools."""

# pylint: disable=missing-function-docstring

import pytest

from device_repository import (
    compare_devices,
    get_device_detail,
    search_devices,
)


def test_search_devices_respects_max_price():
    devices = search_devices(max_price=2000)

    assert len(devices) == 2
    assert all(device["lowest_price_cny"] <= 2000 for device in devices)


def test_search_devices_rejects_negative_price():
    with pytest.raises(ValueError, match="max_price 不能小于 0"):
        search_devices(max_price=-1)


def test_search_devices_rejects_inverted_price_range():
    with pytest.raises(ValueError, match="min_price 不能大于 max_price"):
        search_devices(min_price=3000, max_price=1500)


def test_search_devices_returns_empty_when_nothing_matches():
    devices = search_devices(max_price=100)

    assert devices == []


def test_search_devices_filters_by_brand():
    devices = search_devices(brand="DemoInk")

    assert len(devices) == 2
    assert all(device["brand"] == "DemoInk" for device in devices)


def test_search_devices_combines_brand_max_price():
    devices = search_devices(
        brand="DemoInk",
        max_price=2000,
    )

    assert len(devices) == 1
    assert devices[0]["brand"] == "DemoInk"
    assert devices[0]["model"] == "Reader 6"


def test_search_devices_filters_by_max_weight():
    devices = search_devices(max_weight_g=250)

    assert len(devices) == 2
    assert all(device["weight_g"] <= 250 for device in devices)


def test_search_devices_rejects_non_positive_max_weight():
    with pytest.raises(
        ValueError,
        match="max_weight_g 必须大于 0",
    ):
        search_devices(max_weight_g=0)


def test_get_device_detail_returns_complete_device():

    device = get_device_detail(1)

    assert device is not None
    assert device["id"] == 1
    assert device["brand"] == "DemoInk"
    assert device["model"] == "Reader 6"
    assert device["release_year"] == 2026
    assert device["lowest_price_cny"] == 899


def test_get_device_detail_returns_none_for_unknown_device():
    device = get_device_detail(999_999)

    assert device is None


def test_get_device_detail_rejects_non_positive_id():
    with pytest.raises(
        ValueError,
        match="device_id 必须大于 0",
    ):
        get_device_detail(0)


def test_compare_devices_returns_requested_devices_in_order():
    devices = compare_devices([1, 3])

    assert len(devices) == 2
    assert [device["id"] for device in devices] == [1, 3]
    assert [device["model"] for device in devices] == [
        "Reader 6",
        "Color 7",
    ]


def test_compare_devices_removes_duplicate_ids():
    devices = compare_devices([1, 1, 3])

    assert [device["id"] for device in devices] == [1, 3]


def test_compare_devices_reports_unknown_device():
    with pytest.raises(
        LookupError,
        match="设备不存在:999999",
    ):
        compare_devices([1, 999_999])


def test_get_device_detail_preserves_unknown_fields_as_none():
    device = get_device_detail(1)

    assert device is not None
    assert device["color_ppi"] is None
