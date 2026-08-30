from device_repository import search_devices


def print_results(title: str, devices: list[dict]) -> None:
    print(f"\n{title}")
    print("-" * len(title))

    if not devices:
        print("没有找到符合条件的设备")
        return

    for device in devices:
        color = "彩色" if device["is_color"] else "黑白"
        stylus = "支持手写" if device["supports_stylus"] else "不支持手写"
        price = device["lowest_price_cny"]
        price_text = f"¥{price}" if price is not None else "暂无价格"
        print(
            f"{device['brand']} {device['model']} | "
            f"{device['screen_size_inches']}英寸 | {color} | "
            f"{stylus} | {price_text}"
        )


if __name__ == "__main__":
    print_results(
        "查询一：2000 元以内",
        search_devices(max_price=2000),
    )
    print_results(
        "查询二：8 英寸以上并且支持手写",
        search_devices(min_screen_size=8, supports_stylus=True),
    )
    print_results(
        "查询三：彩色、开放系统、支持手写",
        search_devices(is_color=True, is_open_system=True, supports_stylus=True),
    )
    print_results(
        "查询四： 2000 元以内， 10英寸以下的彩色设备",
        search_devices(max_price=2000, max_screen_size=10, is_color=True),
    )
    print_results(
        "查询五： 1500 到 3000 元之间的设备",
        search_devices(min_price=1500, max_price=3000),
    )
