"""Tests for the tool-calling agent loop."""
import json
from unittest.mock import Mock

import pytest

from eink_agent.agent import run_agent

def _make_tool_call(
    name: str,
    arguments: dict[str, object],
    call_id: str,
) -> Mock:
    function = Mock(
        arguments=json.dumps(arguments),
    )
    function.name = name

    return Mock(
        id=call_id,
        function=function,
    )


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
    assert trace["result_count"] == 2
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

def test_run_agent_executes_device_detail_tool():
    client = Mock()
    tool_call = _make_tool_call(
        "get_device_detail",
        {"device_id": 1},
        "call_detail",
    )

    client.chat.completions.create.side_effect = [
        Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=None,
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="1号设备是 Reader 6。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    result = run_agent(
        "查看1号设备的详情",
        client=client,
        model="test-model",
    )

    trace = result.tool_trace[0]

    assert trace["name"] == "get_device_detail"
    assert trace["arguments"] == {"device_id": 1}
    assert trace["result_count"] == 1
    assert trace["result"]["model"] == "Reader 6"

def test_run_agent_executes_compare_devices_tool():
    client = Mock()
    tool_call = _make_tool_call(
        "compare_devices",
        {"device_ids": [3, 1]},
        "call_compare",
    )

    client.chat.completions.create.side_effect = [
        Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=None,
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="已比较 Color 7 和 Reader 6。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    result = run_agent(
        "比较3号和1号设备",
        client=client,
        model="test-model",
    )

    trace = result.tool_trace[0]

    assert trace["name"] == "compare_devices"
    assert trace["arguments"] == {"device_ids": [3, 1]}
    assert trace["result_count"] == 2
    assert [
        device["model"]
        for device in trace["result"]
    ] == ["Color 7", "Reader 6"]

def test_run_agent_records_zero_results_for_unknown_device():
    client = Mock()
    tool_call = _make_tool_call(
        "get_device_detail",
        {"device_id": 999_999},
        "call_missing",
    )

    client.chat.completions.create.side_effect = [
        Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=None,
                        tool_calls=[tool_call],
                    )
                )
            ]
        ),
        Mock(
            choices=[
                Mock(
                    message=Mock(
                        content="没有找到这个设备。",
                        tool_calls=[],
                    )
                )
            ]
        ),
    ]

    result = run_agent(
        "查看999999号设备",
        client=client,
        model="test-model",
    )

    trace = result.tool_trace[0]

    assert trace["result"] is None
    assert trace["result_count"] == 0


def test_run_agent_sends_unsupported_action_policy():
    client = Mock()
    client.chat.completions.create.return_value = Mock(
        choices=[
            Mock(
                message=Mock(
                    content="我不能替您下单。",
                    tool_calls=[],
                )
            )
        ]
    )

    run_agent(
        "请替我下单1号设备",
        client=client,
        model="test-model",
    )

    request = client.chat.completions.create.call_args.kwargs
    system_content = request["messages"][0]["content"]

    assert "do not call any tool" in system_content
    assert "place orders" in system_content
    assert "modify or delete" in system_content
