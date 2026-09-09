"""Run the tool-calling EInkAgent with DeepSeek."""
import json
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
        result = run_agent(
            user_text,
            client=client,
            model=os.environ["DEEPSEEK_MODEL"],
        )

    print("\n工具执行轨迹：")

    if not result.tool_trace:
        print("- 本次没有调用工具")

    for trace in result.tool_trace:
        arguments = json.dumps(
            trace["arguments"],
            ensure_ascii=False,
        )
        print(f"- 工具：{trace['name']}")
        print(f"  参数：{arguments}")
        print(f"  调用 ID：{trace['tool_call_id']}")
        print(f"  返回记录数：{trace['result_count']}")

    print("\nEInkAgent 回答：")
    print(result.answer)


if __name__ == "__main__":
    main()
