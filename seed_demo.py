from datetime import date

from database import get_connection, initialize_database


# 名称和参数全部是虚构演示数据，只用于学习 SQL。
DEMO_DEVICES = [
    (
        "DemoInk", "Reader 6", "reader", 2026, 6.0, 0, 300, None,
        "DemoOS", 0, 0, 1, 1, 1, 32, 155,
        "适合轻量阅读的虚构演示设备",
    ),
    (
        "DemoInk", "Note 10", "note", 2026, 10.3, 0, 227, None,
        "Android Demo", 1, 1, 1, 1, 0, 64, 430,
        "适合 PDF 和手写的虚构演示设备",
    ),
    (
        "ColorLeaf", "Color 7", "tablet", 2026, 7.0, 1, 300, 150,
        "Android Demo", 1, 1, 1, 1, 0, 128, 245,
        "适合彩色漫画的虚构演示设备",
    ),
]


DEVICE_SQL = """
INSERT OR IGNORE INTO devices (
    brand, model, category, release_year, screen_size_inches,
    is_color, mono_ppi, color_ppi, operating_system, is_open_system,
    supports_stylus, has_front_light, has_warm_light, is_waterproof,
    storage_gb, weight_g, notes
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def seed_demo_data() -> None:
    """Initialize the database and insert the demo reocrds."""
    initialize_database()
    today = date.today().isoformat()

    with get_connection() as connection:
        connection.executemany(DEVICE_SQL, DEMO_DEVICES)

        device_rows = connection.execute(
            "SELECT id, brand, model FROM devices ORDER BY id"
        ).fetchall()

        demo_prices = {
            ("DemoInk", "Reader 6"): 899,
            ("DemoInk", "Note 10"): 2799,
            ("ColorLeaf", "Color 7"): 1899,
        }

        for device in device_rows:
            key = (device["brand"], device["model"])
            if key not in demo_prices:
                continue

            existing = connection.execute(
                """
                SELECT id FROM offers
                WHERE device_id = ? AND seller = '虚构演示商店'
                """,
                (device["id"],),
            ).fetchone()

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO offers (
                        device_id, price_cny, seller, product_url,
                        collected_at, is_available
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        device["id"], demo_prices[key], "虚构演示商店",
                        None, today, 1,
                    ),
                )

if __name__ == "__main__":
    seed_demo_data()
    print("已写入 3 台虚构演示设备和演示价格。")
