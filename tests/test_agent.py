"""Tests for the tool-calling agent loop."""
import pytest

from unittest.mock import Mock

from eink_agent.agent import run_agent


def test_run_agent_executes_search_tool_and_returns_answer():
    client = Mock()

    function = Mock(
        arguments='{"max_price": 2000}',
    )
    function.name = "search_devices"

    tool_call = Mock(
        id="call_123",
        function=function,
    )

    first_message = Mock(
        content=None,
        tool_calls=[tool_call],
    )
    final_message = Mock(
        content="找到两台符合预算的演示设备。",
        tool_calls=[],
    )

    client.chat.completions.create.side_effect = [
        Mock(choices=[Mock(message=first_message)]),
        Mock(choices=[Mock(message=final_message)]),
    ]

    result = run_agent(
        "推荐两千元以内的电子墨水屏设备",
        client=client,
        model="test-model",
    )

    assert result.answer == "找到两台符合预算的演示设备。"
    assert len(result.tool_trace) == 1
    first_request = client.chat.completions.create.call_args_list[0].kwargs
    tool_names = [
        tool["function"]["name"]
        for tool in first_request["tools"]
    ]

    assert tool_names == [
        "search_devices",
        "get_device_detail",
        "compare_devices",
    ]

    trace = result.tool_trace[0]
    assert trace["tool_call_id"] == "call_123"
    assert trace["name"] == "search_devices"
    assert trace["arguments"] == {"max_price": 2000}
    assert len(trace["result"]) == 2

    assert client.chat.completions.create.call_count == 2

    second_request = client.chat.completions.create.call_args_list[1].kwargs
    tool_message = second_request["messages"][-1]

    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_123"
    assert '"Reader 6"' in tool_message["content"]

def test_run_agent_returns_direct_model_answer():
    client = Mock()
    client.chat.completions.create.return_value = Mock(
        choices=[
            Mock(
                message=Mock(
                    content="请先告诉我预算和主要用途。",
                    tool_calls=[],
                )
            )
        ]
    )

    result = run_agent(
        "我想买电子墨水屏",
        client=client,
        model="test-model",
    )

    assert result.answer == "请先告诉我预算和主要用途。"
    assert result.tool_trace == []

def test_run_agent_stops_after_maximum_rounds():
    client = Mock()

    function = Mock(arguments="{}")
    function.name = "search_devices"

    repeated_tool_call = Mock(
        id="call_repeated",
        function=function,
    )
    repeated_message = Mock(
        content=None,
        tool_calls=[repeated_tool_call],
    )

    client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=repeated_message)]
    )

    with pytest.raises(RuntimeError, match="maximum rounds"):
        run_agent(
            "一直搜索设备",
            client=client,
            model="test-model",
            max_rounds=2,
        )

    assert client.chat.completions.create.call_count == 2
