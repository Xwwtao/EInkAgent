from database import get_connection


QUERY = """
SELECT
    d.id, d.brand, d.model, d.screen_size_inches,
    d.is_color, d.supports_stylus,
    MIN(o.price_cny) AS lowest_price_cny
FROM devices AS d
LEFT JOIN offers AS o
    ON o.device_id = d.id AND o.is_available = 1
GROUP BY d.id
ORDER BY d.id
"""


if __name__ == "__main__":
    with get_connection() as connection:
        devices = connection.execute(QUERY).fetchall()

    if not devices:
        print("数据库里还没有设备，请先运行 python3 seed_demo.py")
    else:
        for device in devices:
            color = "彩色" if device["is_color"] else "黑白"
            stylus = "支持手写" if device["supports_stylus"] else "不支持手写"
            price = device["lowest_price_cny"]
            price_text = f"¥{price}" if price is not None else "暂无价格"
            print(
                f"#{device['id']} {device['brand']} {device['model']} | "
                f"{device['screen_size_inches']}英寸 | {color} | "
                f"{stylus} | {price_text}"
            )
