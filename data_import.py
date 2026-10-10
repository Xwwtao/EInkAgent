"""Import real device and offer data from CSV into SQLite.

The CSV files in ``data/`` are the source of truth and are version controlled,
while the SQLite database itself is not. Re-running the import must never
duplicate rows, so every write below is keyed on the natural key of the record.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import csv
from pathlib import Path
from typing import Any

from database import get_connection, initialize_database

PROJECT_DIR = Path(__file__).resolve().parent
DEVICES_CSV = PROJECT_DIR / "data" / "real_devices.csv"
OFFERS_CSV = PROJECT_DIR / "data" / "real_offers.csv"
EXCHANGE_RATES_CSV = PROJECT_DIR / "data" / "exchange_rates.csv"

EXCHANGE_RATE_COLUMNS = ("currency", "rate_to_cny", "as_of", "source_note")

# CNY is the unit of price_cny, so it never needs an entry in the rate file.
BASE_CURRENCY = "CNY"

DEVICE_COLUMNS = (
    "brand",
    "model",
    "category",
    "release_year",
    "screen_size_inches",
    "is_color",
    "mono_ppi",
    "color_ppi",
    "operating_system",
    "is_open_system",
    "supports_stylus",
    "has_front_light",
    "has_warm_light",
    "is_waterproof",
    "storage_gb",
    "weight_g",
    "notes",
    "spec_source_url",
    "spec_verified_at",
)

OFFER_COLUMNS = (
    "brand",
    "model",
    "seller",
    "price_amount",
    "currency",
    "product_url",
    "collected_at",
    "is_available",
)

CATEGORIES = ("reader", "note", "tablet")

INTEGER_COLUMNS = (
    "release_year",
    "mono_ppi",
    "color_ppi",
    "storage_gb",
    "weight_g",
)

FLAG_COLUMNS = (
    "is_color",
    "is_open_system",
    "supports_stylus",
    "has_front_light",
    "has_warm_light",
    "is_waterproof",
)

POSITIVE_INTEGER_COLUMNS = (
    "release_year",
    "mono_ppi",
    "color_ppi",
    "storage_gb",
    "weight_g",
)

# Fields that must be present for a real record to be trustworthy.
REQUIRED_DEVICE_FIELDS = (
    "brand",
    "model",
    "category",
    "screen_size_inches",
    "spec_source_url",
    "spec_verified_at",
)

REQUIRED_OFFER_FIELDS = (
    "brand",
    "model",
    "seller",
    "price_amount",
    "currency",
    "product_url",
    "collected_at",
    "is_available",
)


class DataValidationError(ValueError):
    """Raised when a CSV row fails validation."""


@dataclass(frozen=True)
class DeviceRecord:
    """A validated device row, ready to be written to SQLite."""

    values: dict[str, Any]
    unpublished_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class OfferRecord:
    """A validated offer row, ready to be written to SQLite."""

    brand: str
    model: str
    price_cny: int
    price_amount_minor: int
    currency: str
    seller: str
    product_url: str
    collected_at: str
    is_available: int


def _normalize(value: Any) -> str:
    """Turn a raw CSV cell into a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def _parse_integer(
    field: str,
    raw: str,
    *,
    positive: bool = False,
) -> int | None:
    """Parse an optional integer cell.

    Values that were not published stay empty on purpose: the data principle is
    that unknown parameters are stored as NULL rather than guessed.
    """
    if not raw:
        return None

    try:
        number = int(raw)
    except ValueError as error:
        raise DataValidationError(f"{field} 必须是整数，收到 {raw!r}") from error

    if positive and number <= 0:
        raise DataValidationError(f"{field} 必须大于 0，收到 {number}")

    return number


def _parse_float(field: str, raw: str) -> float | None:
    """Parse an optional float cell."""
    if not raw:
        return None

    try:
        number = float(raw)
    except ValueError as error:
        raise DataValidationError(f"{field} 必须是数字，收到 {raw!r}") from error

    if number <= 0:
        raise DataValidationError(f"{field} 必须大于 0，收到 {number}")

    return number


def _parse_optional_flag(field: str, raw: str) -> int | None:
    """Parse an optional 0/1 flag; blank means unknown."""
    if not raw:
        return None

    if raw not in {"0", "1"}:
        raise DataValidationError(f"{field} 只能是 0 或 1，收到 {raw!r}")

    return int(raw)


def _parse_required_flag(field: str, raw: str) -> int:
    """Parse a required 0/1 flag."""
    value = _parse_optional_flag(field, raw)

    if value is None:
        raise DataValidationError(f"{field} 不能为空")

    return value


def _parse_iso_date(field: str, raw: str) -> str:
    """Parse a required ISO ``YYYY-MM-DD`` cell."""
    try:
        return date.fromisoformat(raw).isoformat()
    except ValueError as error:
        raise DataValidationError(
            f"{field} 必须是 YYYY-MM-DD 格式，收到 {raw!r}"
        ) from error


def _parse_required_url(field: str, raw: str) -> str:
    """Parse a required http(s) URL cell."""
    if not raw.startswith(("http://", "https://")):
        raise DataValidationError(f"{field} 必须是 http(s) 链接，收到 {raw!r}")

    return raw


def _parse_price_amount(raw: str) -> Decimal:
    """Parse an original price with at most two decimal places."""
    try:
        amount = Decimal(raw)
    except InvalidOperation as error:
        raise DataValidationError(
            f"price_amount 必须是数字，收到 {raw!r}"
        ) from error

    if not amount.is_finite() or amount < 0:
        raise DataValidationError(f"price_amount 必须大于等于 0，收到 {raw!r}")

    if amount != amount.quantize(Decimal("0.01")):
        raise DataValidationError(f"price_amount 最多两位小数，收到 {raw!r}")

    return amount


def _rate_to_cny(currency: str, rates: dict[str, Decimal]) -> Decimal:
    """Return the CNY rate for a currency registered in the rate table."""
    if currency == BASE_CURRENCY:
        return Decimal(1)

    if currency not in rates:
        raise DataValidationError(
            f"currency {currency!r} 不在汇率表中，可选："
            f"{', '.join([BASE_CURRENCY, *sorted(rates)])}"
        )

    return rates[currency]


def load_exchange_rates(path: Path = EXCHANGE_RATES_CSV) -> dict[str, Decimal]:
    """Read the fixed exchange rates used to derive price_cny."""
    rates: dict[str, Decimal] = {}

    for line, row in enumerate(read_csv_rows(path, EXCHANGE_RATE_COLUMNS), start=2):
        try:
            rate = Decimal(row["rate_to_cny"])
        except InvalidOperation as error:
            raise DataValidationError(
                f"第 {line} 行 rate_to_cny 必须是数字，收到 {row['rate_to_cny']!r}"
            ) from error

        if not rate.is_finite() or rate <= 0:
            raise DataValidationError(
                f"第 {line} 行 rate_to_cny 必须大于 0，收到 {row['rate_to_cny']!r}"
            )

        rates[row["currency"]] = rate

    return rates


def read_csv_rows(path: Path, expected_columns: Iterable[str]) -> list[dict[str, str]]:
    """Read a CSV file and verify its header."""
    expected = tuple(expected_columns)

    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())

        if header != expected:
            raise DataValidationError(
                f"{path.name} 表头不匹配。期望 {expected}，收到 {header}"
            )

        return [
            {key: _normalize(value) for key, value in row.items()}
            for row in reader
        ]


def validate_device_row(row: dict[str, str], line: int) -> DeviceRecord:
    """Validate one device row and return the SQLite-ready values."""
    missing = [
        field
        for field in REQUIRED_DEVICE_FIELDS
        if not row.get(field)
    ]

    if missing:
        raise DataValidationError(f"第 {line} 行缺少必填字段：{', '.join(missing)}")

    category = row["category"]
    if category not in CATEGORIES:
        raise DataValidationError(
            f"第 {line} 行 category 必须是 {CATEGORIES} 之一，收到 {category!r}"
        )

    unpublished_flags: list[str] = []

    values: dict[str, Any] = {
        "brand": row["brand"],
        "model": row["model"],
        "category": category,
        "screen_size_inches": _parse_float(
            "screen_size_inches", row["screen_size_inches"]
        ),
        "operating_system": row["operating_system"] or None,
        "notes": row["notes"] or None,
        "spec_source_url": _parse_required_url(
            "spec_source_url", row["spec_source_url"]
        ),
        "spec_verified_at": _parse_iso_date(
            "spec_verified_at", row["spec_verified_at"]
        ),
    }

    for field in INTEGER_COLUMNS:
        values[field] = _parse_integer(
            field,
            row[field],
            positive=field in POSITIVE_INTEGER_COLUMNS,
        )

    for field in FLAG_COLUMNS:
        if not row[field]:
            unpublished_flags.append(field)
        values[field] = _parse_optional_flag(field, row[field])

    return DeviceRecord(values=values, unpublished_flags=tuple(unpublished_flags))


def validate_offer_row(
    row: dict[str, str],
    line: int,
    rates: dict[str, Decimal],
) -> OfferRecord:
    """Validate one offer row and derive its CNY reference price."""
    missing = [
        field
        for field in REQUIRED_OFFER_FIELDS
        if not row.get(field)
    ]

    if missing:
        raise DataValidationError(f"第 {line} 行缺少必填字段：{', '.join(missing)}")

    try:
        amount = _parse_price_amount(row["price_amount"])
        rate = _rate_to_cny(row["currency"], rates)
    except DataValidationError as error:
        raise DataValidationError(f"第 {line} 行 {error}") from error

    price_cny = int((amount * rate).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    is_available = _parse_required_flag("is_available", row["is_available"])

    return OfferRecord(
        brand=row["brand"],
        model=row["model"],
        price_cny=price_cny,
        price_amount_minor=int(amount * 100),
        currency=row["currency"],
        seller=row["seller"],
        product_url=_parse_required_url("product_url", row["product_url"]),
        collected_at=_parse_iso_date("collected_at", row["collected_at"]),
        is_available=is_available,
    )


def _device_id(connection: Any, brand: str, model: str) -> int | None:
    """Return the id for a brand and model pair, if it exists."""
    row = connection.execute(
        "SELECT id FROM devices WHERE brand = ? AND model = ?",
        (brand, model),
    ).fetchone()

    return None if row is None else int(row["id"])


def import_devices(
    rows: list[dict[str, str]],
) -> tuple[int, int, dict[str, list[str]]]:
    """Insert or update device rows.

    Returns (inserted, updated, unpublished_flags), where the last item maps a
    device name to the flag fields that were blank in the CSV and therefore
    stored as NULL (unknown), see docs/decisions/002-nullable-device-flags.md.
    """
    inserted = 0
    updated = 0
    unpublished: dict[str, list[str]] = {}

    initialize_database()

    with get_connection() as connection:
        for line, row in enumerate(rows, start=2):
            record = validate_device_row(row, line)
            values = record.values

            if record.unpublished_flags:
                unpublished[f"{values['brand']} {values['model']}"] = list(
                    record.unpublished_flags
                )

            existing_id = _device_id(
                connection, str(values["brand"]), str(values["model"])
            )

            columns = tuple(values)
            placeholders = ", ".join("?" for _ in columns)
            assignments = ", ".join(f"{name} = ?" for name in columns)

            if existing_id is None:
                connection.execute(
                    f"INSERT INTO devices ({', '.join(columns)}) "
                    f"VALUES ({placeholders})",
                    [values[name] for name in columns],
                )
                inserted += 1
            else:
                connection.execute(
                    f"UPDATE devices SET {assignments} WHERE id = ?",
                    [values[name] for name in columns] + [existing_id],
                )
                updated += 1

    return inserted, updated, unpublished


def import_offers(
    rows: list[dict[str, str]],
    *,
    rates: dict[str, Decimal],
) -> tuple[int, int, list[str]]:
    """Insert offer rows that are not present yet.

    ``rates`` maps a currency code to its fixed CNY rate; it is passed in
    rather than read here so tests never depend on the real rate file.

    Returns (inserted, skipped, unresolved), where ``unresolved`` lists rows
    whose device is missing from the database.
    """
    inserted = 0
    skipped = 0
    unresolved: list[str] = []

    with get_connection() as connection:
        for line, row in enumerate(rows, start=2):
            record = validate_offer_row(row, line, rates)

            device_id = _device_id(connection, record.brand, record.model)
            if device_id is None:
                unresolved.append(f"第 {line} 行：{record.brand} {record.model}")
                continue

            existing = connection.execute(
                """
                SELECT id FROM offers
                WHERE device_id = ? AND seller = ? AND collected_at = ?
                """,
                (device_id, record.seller, record.collected_at),
            ).fetchone()

            if existing is not None:
                skipped += 1
                continue

            connection.execute(
                """
                INSERT INTO offers (
                    device_id, price_cny, price_amount_minor, currency,
                    seller, product_url, collected_at, is_available
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device_id,
                    record.price_cny,
                    record.price_amount_minor,
                    record.currency,
                    record.seller,
                    record.product_url,
                    record.collected_at,
                    record.is_available,
                ),
            )
            inserted += 1

    return inserted, skipped, unresolved


def main() -> None:
    """Import both CSV files and print a short summary."""
    device_rows = read_csv_rows(DEVICES_CSV, DEVICE_COLUMNS)
    offer_rows = read_csv_rows(OFFERS_CSV, OFFER_COLUMNS)
    rates = load_exchange_rates()

    inserted_devices, updated_devices, unpublished = import_devices(device_rows)
    inserted_offers, skipped_offers, unresolved = import_offers(offer_rows, rates=rates)

    print(
        f"设备：新增 {inserted_devices} 台，更新 {updated_devices} 台"
        f"（共读取 {len(device_rows)} 行）"
    )
    print(
        f"报价：新增 {inserted_offers} 条，跳过已存在 {skipped_offers} 条"
        f"（共读取 {len(offer_rows)} 行）"
    )

    if unpublished:
        print(
            f"以下设备的标志位在 CSV 中留空，已按 NULL（未知）写入，"
            f"需人工核验：共 {len(unpublished)} 台"
        )
        for name, fields in sorted(unpublished.items()):
            print(f"  - {name}: {', '.join(fields)}")

    if unresolved:
        print("以下报价找不到对应设备，未导入：")
        for item in unresolved:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
