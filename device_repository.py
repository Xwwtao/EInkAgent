"""Query tools for searching, inspecting, and comparing E Ink devices."""

from typing import Any

from database import get_connection

# The cheapest available offer per device, with where and when it was seen.
# Equal prices prefer the most recent snapshot.
LOWEST_PRICE_JOIN = """
LEFT JOIN (
    SELECT
        device_id,
        price_cny AS lowest_price_cny,
        seller AS price_seller,
        collected_at AS price_collected_at,
        product_url AS price_url,
        ROW_NUMBER() OVER (
            PARTITION BY device_id
            ORDER BY price_cny, collected_at DESC, id
        ) AS price_rank
    FROM offers
    WHERE is_available = 1
) AS prices ON prices.device_id = d.id AND prices.price_rank = 1
"""

PRICE_COLUMNS = """
    prices.lowest_price_cny, prices.price_seller,
    prices.price_collected_at, prices.price_url
"""

BASE_QUERY = f"""
SELECT
    d.id, d.brand, d.model, d.category, d.screen_size_inches,
    d.is_color, d.is_open_system, d.supports_stylus,
    d.storage_gb, d.weight_g, {PRICE_COLUMNS}
FROM devices AS d
{LOWEST_PRICE_JOIN}
"""

DETAIL_QUERY = f"""
SELECT
    d.*, {PRICE_COLUMNS}
FROM devices AS d
{LOWEST_PRICE_JOIN}
WHERE d.id = ?
"""


def _escape_like(text: str) -> str:
    """Make %, _ and the escape character match literally in LIKE."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_devices(
    *,
    max_price: int | None = None,
    min_price: int | None = None,
    min_screen_size: float | None = None,
    max_screen_size: float | None = None,
    max_weight_g: int | None = None,
    supports_stylus: bool | None = None,
    is_color: bool | None = None,
    is_open_system: bool | None = None,
    category: str | None = None,
    brand: str | None = None,
    keyword: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """根据选购条件查询设备；None 表示用户未限制该条件。"""
    if max_weight_g is not None and max_weight_g <= 0:
        raise ValueError("max_weight_g 必须大于 0")
    if max_price is not None and max_price < 0:
        raise ValueError("max_price 不能小于 0")
    if min_price is not None and min_price < 0:
        raise ValueError("min_price 不能小于 0")
    if min_price is not None and max_price is not None and min_price > max_price:
        raise ValueError("min_price 不能大于 max_price")
    if min_screen_size is not None and min_screen_size <= 0:
        raise ValueError("min_screen_size 必须大于 0")
    if max_screen_size is not None and max_screen_size <= 0:
        raise ValueError("max_screen_size 必须大于 0")
    if not 1 <= limit <= 100:
        raise ValueError("limit 必须在 1 到 100 之间")
    if keyword is not None and not keyword.split():
        raise ValueError("keyword 不能为空白")

    conditions: list[str] = []
    parameters: list[Any] = []

    if brand is not None:
        conditions.append("d.brand = ?")
        parameters.append(brand)
    if keyword is not None:
        # Every whitespace-separated token must appear in "brand model", so
        # "Kindle Paperwhite 2026" matches "Kindle Paperwhite (2026)".
        for token in keyword.split():
            conditions.append(
                "LOWER(d.brand || ' ' || d.model) LIKE ? ESCAPE '\\'"
            )
            parameters.append(f"%{_escape_like(token.lower())}%")
    if max_weight_g is not None:
        conditions.append("d.weight_g <= ?")
        parameters.append(max_weight_g)
    if max_price is not None:
        conditions.append("prices.lowest_price_cny <= ?")
        parameters.append(max_price)
    if min_price is not None:
        conditions.append("prices.lowest_price_cny >= ?")
        parameters.append(min_price)
    if min_screen_size is not None:
        conditions.append("d.screen_size_inches >= ?")
        parameters.append(min_screen_size)
    if max_screen_size is not None:
        conditions.append("d.screen_size_inches <= ?")
        parameters.append(max_screen_size)
    if supports_stylus is not None:
        conditions.append("d.supports_stylus = ?")
        parameters.append(int(supports_stylus))
    if is_color is not None:
        conditions.append("d.is_color = ?")
        parameters.append(int(is_color))
    if is_open_system is not None:
        conditions.append("d.is_open_system = ?")
        parameters.append(int(is_open_system))
    if category is not None:
        conditions.append("d.category = ?")
        parameters.append(category)

    query = BASE_QUERY
    if conditions:
        query += "\nWHERE " + " AND ".join(conditions)

    query += "\nORDER BY prices.lowest_price_cny IS NULL, prices.lowest_price_cny, d.id"
    query += "\nLIMIT ?"
    parameters.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]


def get_device_detail(
    device_id: int,
) -> dict[str, Any] | None:
    """ "根据设备ID返回完整详情； 不存在时返回None。"""
    if device_id <= 0:
        raise ValueError("device_id 必须大于 0")

    with get_connection() as connection:
        row = connection.execute(
            DETAIL_QUERY,
            (device_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def compare_devices(
    device_ids: list[int],
) -> list[dict[str, Any]]:
    """按输入顺序返回存在的设备详情。"""
    devices: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for device_id in device_ids:
        if device_id in seen_ids:
            continue

        seen_ids.add(device_id)

        device = get_device_detail(device_id)
        if device is None:
            raise LookupError(f"设备不存在:{device_id}")

        devices.append(device)

    return devices
