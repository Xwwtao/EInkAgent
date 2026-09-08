"""Run the tool-calling EInkAgent with DeepSeek."""

import os

from openai import OpenAI

from eink_agent.agent import run_agent
from seed_demo import seed_demo_data


def main() -> None:
    """Run one interactive Agent request."""
    user_text = input("请输入选购需求：").strip()

    if not user_text:
        print("需求不能为空。")
        return

    seed_demo_data()

    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=0,
    ) as client:
        answer = run_agent(
            user_text,
            client=client,
            model=os.environ["DEEPSEEK_MODEL"],
        )

    print("\nEInkAgent 回答：")
    print(answer)


if __name__ == "__main__":
    main()