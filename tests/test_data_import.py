"""Tests for the real-data importer and the data quality report."""

from datetime import date
from decimal import Decimal
import csv

import pytest

from data_import import (
    DEVICE_COLUMNS,
    OFFER_COLUMNS,
    DataValidationError,
    import_devices,
    import_offers,
    load_exchange_rates,
    read_csv_rows,
)
from database import get_connection
from data_quality import build_report
from device_repository import search_devices


# 测试使用固定汇率，不读取 data/exchange_rates.csv。
RATES = {"USD": Decimal("7.1")}


def _device_row(**overrides: str) -> dict[str, str]:
    """Return a minimal valid device row, with optional overrides."""
    row = {column: "" for column in DEVICE_COLUMNS}
    row.update(
        {
            "brand": "BOOX",
            "model": "Note Air3 C",
            "category": "note",
            "release_year": "2023",
            "screen_size_inches": "10.3",
            "is_color": "1",
            "mono_ppi": "227",
            "color_ppi": "150",
            "operating_system": "Android 11",
            "is_open_system": "1",
            "supports_stylus": "1",
            "has_front_light": "1",
            "has_warm_light": "1",
            "is_waterproof": "0",
            "storage_gb": "64",
            "weight_g": "430",
            "notes": "10.3吋彩色手写本",
            "spec_source_url": "https://example.com/boox-note-air3-c",
            "spec_verified_at": "2026-09-11",
        }
    )
    row.update(overrides)

    return row


def _offer_row(**overrides: str) -> dict[str, str]:
    """Return a minimal valid offer row, with optional overrides."""
    row = {column: "" for column in OFFER_COLUMNS}
    row.update(
        {
            "brand": "BOOX",
            "model": "Note Air3 C",
            "seller": "官方旗舰店",
            "price_amount": "2980",
            "currency": "CNY",
            "product_url": "https://example.com/offer/1",
            "collected_at": "2026-09-11",
            "is_available": "1",
        }
    )
    row.update(overrides)

    return row


def _device_count() -> int:
    with get_connection() as connection:
        return int(
            connection.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
        )


def _offer_count() -> int:
    with get_connection() as connection:
        return int(
            connection.execute("SELECT COUNT(*) FROM offers").fetchone()[0]
        )


def test_import_devices_inserts_real_record_with_unknown_fields_as_null():
    import_devices([_device_row(mono_ppi="", color_ppi="")])

    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM devices WHERE brand = ? AND model = ?",
            ("BOOX", "Note Air3 C"),
        ).fetchone()

    assert row is not None
    assert row["category"] == "note"
    assert row["is_color"] == 1
    assert row["weight_g"] == 430
    assert row["mono_ppi"] is None
    assert row["spec_source_url"] == "https://example.com/boox-note-air3-c"
    assert row["spec_verified_at"] == "2026-09-11"


def test_import_devices_is_idempotent():
    import_devices([_device_row()])
    after_first = _device_count()

    import_devices([_device_row()])

    assert _device_count() == after_first


def test_import_devices_updates_changed_value_without_duplicating():
    import_devices([_device_row(weight_g="430")])
    import_devices([_device_row(weight_g="435")])

    with get_connection() as connection:
        rows = connection.execute(
            "SELECT weight_g FROM devices WHERE brand = ? AND model = ?",
            ("BOOX", "Note Air3 C"),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["weight_g"] == 435


def test_import_devices_keeps_demo_data_intact():
    """真实数据导入不得影响演示设备（测试 fixture 依赖它们）。"""
    import_devices([_device_row()])

    with get_connection() as connection:
        demo = connection.execute(
            "SELECT COUNT(*) FROM devices WHERE brand = 'DemoInk'"
        ).fetchone()[0]

    assert demo == 2


def test_import_devices_reports_no_unpublished_flags_when_complete():
    _, _, unpublished = import_devices([_device_row()])

    assert unpublished == {}


def test_import_devices_rejects_row_without_source():
    with pytest.raises(DataValidationError, match="spec_source_url"):
        import_devices([_device_row(spec_source_url="")])


def test_import_devices_rejects_unknown_category():
    with pytest.raises(DataValidationError, match="category"):
        import_devices([_device_row(category="phone")])


def test_import_devices_rejects_invalid_date():
    with pytest.raises(DataValidationError, match="spec_verified_at"):
        import_devices([_device_row(spec_verified_at="2026/09/11")])


def test_import_devices_rejects_non_boolean_flag():
    with pytest.raises(DataValidationError, match="is_color"):
        import_devices([_device_row(is_color="yes")])


def test_import_devices_rejects_negative_weight():
    with pytest.raises(DataValidationError, match="weight_g"):
        import_devices([_device_row(weight_g="-10")])


def test_import_offers_is_idempotent_and_links_device():
    import_devices([_device_row()])

    first = import_offers([_offer_row()], rates=RATES)
    second = import_offers([_offer_row()], rates=RATES)

    assert first[0] == 1
    assert first[1] == 0
    assert second[0] == 0
    assert second[1] == 1

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT offers.price_cny, devices.brand
            FROM offers
            JOIN devices ON devices.id = offers.device_id
            WHERE devices.brand = ?
            """,
            ("BOOX",),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["price_cny"] == 2980


def test_import_offers_reports_unresolved_device():
    imported, skipped, unresolved = import_offers([_offer_row()], rates=RATES)

    assert (imported, skipped) == (0, 0)
    assert len(unresolved) == 1
    assert "BOOX Note Air3 C" in unresolved[0]


def test_import_offers_rejects_missing_url():
    import_devices([_device_row()])

    with pytest.raises(DataValidationError, match="product_url"):
        import_offers([_offer_row(product_url="")], rates=RATES)


def test_read_csv_rows_parses_device_file(tmp_path):
    path = tmp_path / "devices.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEVICE_COLUMNS))
        writer.writeheader()
        writer.writerow(_device_row())

    rows = read_csv_rows(path, DEVICE_COLUMNS)

    assert len(rows) == 1
    assert rows[0]["brand"] == "BOOX"
    assert rows[0]["color_ppi"] == "150"


def test_read_csv_rows_rejects_wrong_header(tmp_path):
    path = tmp_path / "devices.csv"
    path.write_text("brand,model\nBOOX,Note Air3 C\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="表头不匹配"):
        read_csv_rows(path, DEVICE_COLUMNS)


def test_import_devices_stores_unknown_flag_as_null_and_reports_it():
    _, _, unpublished = import_devices([_device_row(has_warm_light="")])

    assert unpublished == {"BOOX Note Air3 C": ["has_warm_light"]}

    with get_connection() as connection:
        value = connection.execute(
            """
            SELECT has_warm_light
            FROM devices
            WHERE brand = ? AND model = ?
            """,
            ("BOOX", "Note Air3 C"),
        ).fetchone()[0]

    assert value is None


def test_import_offers_rejects_missing_availability():
    import_devices([_device_row()])

    with pytest.raises(DataValidationError, match="is_available"):
        import_offers([_offer_row(is_available="")], rates=RATES)


def test_unknown_open_system_is_not_treated_as_closed():
    import_devices([_device_row(is_open_system="")])

    results = search_devices(is_open_system=False)

    assert "Note Air3 C" not in {item["model"] for item in results}


def test_report_with_only_demo_data_counts_zero_real_records():
    """conftest 只写入 3 台演示设备和 3 条演示报价，真实数据应全部为 0。"""
    report = build_report(today=date(2026, 9, 11))

    assert "设备：真实 0 台 / 演示 3 台 / 合计 3 台" in report
    assert "报价：0 条，覆盖 0 台设备" in report
    assert "- [ ] 真实设备 ≥ 20 台（当前 0）" in report
    assert "- [ ] 报价 ≥ 40 条（当前 0）" in report
    assert "- [ ] 来源与核验时间 100% 非空" in report


def test_report_category_distribution_excludes_demo_devices():
    import_devices([_device_row()])

    report = build_report(today=date(2026, 9, 11))

    assert "- note：1 台" in report
    assert "- reader" not in report
    assert "- tablet" not in report


def test_report_counts_real_offers_and_stale_snapshots():
    import_devices([_device_row()])
    import_offers(
        [
            _offer_row(collected_at="2026-09-11"),
            _offer_row(seller="第二卖家", collected_at="2026-07-01"),
        ],
        rates=RATES,
    )

    report = build_report(today=date(2026, 9, 11))

    assert "报价：2 条，覆盖 1 台设备" in report
    assert "采集时间超过 30 天的报价：1 条" in report
    assert "没有任何报价的真实设备：0 台" in report


def _stored_offer() -> dict:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT price_cny, price_amount_minor, currency
            FROM offers
            JOIN devices ON devices.id = offers.device_id
            WHERE devices.brand = ?
            """,
            ("BOOX",),
        ).fetchone()

    return dict(row)


def test_import_offers_converts_usd_and_keeps_original_price():
    import_devices([_device_row()])

    import_offers([_offer_row(price_amount="629", currency="USD")], rates=RATES)

    assert _stored_offer() == {
        "price_cny": 4466,
        "price_amount_minor": 62900,
        "currency": "USD",
    }


def test_import_offers_rounds_converted_price_half_up():
    import_devices([_device_row()])

    # 109.99 × 7.1 = 780.929 → 781
    import_offers([_offer_row(price_amount="109.99", currency="USD")], rates=RATES)

    assert _stored_offer()["price_cny"] == 781
    assert _stored_offer()["price_amount_minor"] == 10999


def test_import_offers_keeps_cny_price_unchanged():
    import_devices([_device_row()])

    import_offers([_offer_row(price_amount="2980", currency="CNY")], rates=RATES)

    assert _stored_offer() == {
        "price_cny": 2980,
        "price_amount_minor": 298000,
        "currency": "CNY",
    }


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"price_amount": ""}, "price_amount"),
        ({"price_amount": "abc"}, "price_amount"),
        ({"price_amount": "-1"}, "price_amount"),
        ({"price_amount": "1.999"}, "price_amount"),
        ({"currency": ""}, "currency"),
        ({"currency": "usd"}, "currency"),
        ({"currency": "EUR"}, "currency"),
    ],
)
def test_import_offers_rejects_invalid_price_or_currency(overrides, field):
    import_devices([_device_row()])

    with pytest.raises(DataValidationError, match=field):
        import_offers([_offer_row(**overrides)], rates=RATES)


def test_usd_offer_is_not_matched_by_cny_budget_as_raw_amount():
    """$629 必须按换算后的人民币参与预算筛选，而不是当作 629 元。"""
    import_devices([_device_row()])
    import_offers([_offer_row(price_amount="629", currency="USD")], rates=RATES)

    results = search_devices(max_price=2000)

    assert "Note Air3 C" not in {item["model"] for item in results}


def test_load_exchange_rates_reads_rate_file(tmp_path):
    path = tmp_path / "rates.csv"
    path.write_text(
        "currency,rate_to_cny,as_of,source_note\nUSD,7.1,2026-09-11,固定\n",
        encoding="utf-8",
    )

    assert load_exchange_rates(path) == {"USD": Decimal("7.1")}


@pytest.mark.parametrize("rate", ["0", "-7.1", "abc", ""])
def test_load_exchange_rates_rejects_invalid_rate(tmp_path, rate):
    path = tmp_path / "rates.csv"
    path.write_text(
        f"currency,rate_to_cny,as_of,source_note\nUSD,{rate},2026-09-11,固定\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="rate_to_cny"):
        load_exchange_rates(path)


def test_schema_rejects_duplicate_offer_without_running_device_import():
    """唯一索引必须由 schema.sql 创建，而不是依赖先运行 import_devices。"""
    import sqlite3

    with get_connection() as connection:
        device_id = connection.execute(
            "SELECT id FROM devices WHERE brand = 'DemoInk' ORDER BY id"
        ).fetchone()[0]

        insert = """
            INSERT INTO offers (
                device_id, price_cny, price_amount_minor, currency,
                seller, product_url, collected_at, is_available
            ) VALUES (?, 100, 10000, 'CNY', '重复卖家', NULL, '2026-10-01', 1)
        """
        connection.execute(insert, (device_id,))

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(insert, (device_id,))
