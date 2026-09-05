"""Demonstrate natural-language device search with DeepSeek."""

import os

from openai import OpenAI

from device_repository import search_devices
from eink_agent.requirement_parser import parse_requirements


def main() -> None:
    user_text = input("请输入选购需求：").strip()
    if not user_text:
        print("需求不能为空。")
        return

    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    ) as client:
        requirements = parse_requirements(
            user_text,
            client=client,
            model=os.environ["DEEPSEEK_MODEL"],
        )

    print("解析出的条件：")
    print(requirements.model_dump_json(indent=2, exclude_none=True))

    devices = search_devices(
        **requirements.model_dump(exclude_none=True)
    )

    print(f"找到 {len(devices)} 台演示设备：")
    for device in devices:
        print(device["brand"], device["model"], device["lowest_price_cny"])


if __name__ == "__main__":
    main()