"""Behavior tests for the E Ink device repository tools."""

# pylint: disable=missing-function-docstring

import pytest

from database import get_connection

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


def test_search_devices_keyword_matches_brand_and_model_tokens():
    results = search_devices(keyword="demoink reader")

    assert [device["model"] for device in results] == ["Reader 6"]


def test_search_devices_keyword_tokens_ignore_punctuation_between_words():
    results = search_devices(keyword="Color 7")

    assert [device["model"] for device in results] == ["Color 7"]


def test_search_devices_keyword_without_match_returns_empty_list():
    assert search_devices(keyword="Paperwhite") == []


def test_search_devices_keyword_treats_like_wildcards_literally():
    assert search_devices(keyword="%") == []
    assert search_devices(keyword="_") == []


def test_search_devices_rejects_blank_keyword():
    with pytest.raises(ValueError, match="keyword"):
        search_devices(keyword="   ")


def _add_offer(model: str, price: int, seller: str, collected_at: str, available: int = 1) -> None:
    with get_connection() as connection:
        device_id = connection.execute(
            "SELECT id FROM devices WHERE model = ?", (model,)
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO offers (
                device_id, price_cny, price_amount_minor, currency,
                seller, product_url, collected_at, is_available
            ) VALUES (?, ?, ?, 'CNY', ?, ?, ?, ?)
            """,
            (device_id, price, price * 100, seller, f"https://example.com/{seller}", collected_at, available),
        )


def test_search_results_include_where_the_lowest_price_came_from():
    _add_offer("Reader 6", 799, "低价卖家", "2026-10-01")

    device = search_devices(keyword="Reader 6")[0]

    assert device["lowest_price_cny"] == 799
    assert device["price_seller"] == "低价卖家"
    assert device["price_collected_at"] == "2026-10-01"
    assert device["price_url"] == "https://example.com/低价卖家"


def test_lowest_price_ignores_unavailable_offers():
    _add_offer("Reader 6", 1, "缺货卖家", "2026-10-01", available=0)

    device = search_devices(keyword="Reader 6")[0]

    assert device["lowest_price_cny"] == 899
    assert device["price_seller"] == "虚构演示商店"


def test_equal_lowest_prices_use_the_most_recent_snapshot():
    _add_offer("Reader 6", 899, "新卖家", "2099-01-01")

    device = get_device_detail(search_devices(keyword="Reader 6")[0]["id"])

    assert device["price_seller"] == "新卖家"
    assert device["price_collected_at"] == "2099-01-01"


def test_device_without_offers_has_no_price_source():
    with get_connection() as connection:
        connection.execute("DELETE FROM offers")

    device = search_devices(keyword="Reader 6")[0]

    assert device["lowest_price_cny"] is None
    assert device["price_seller"] is None
