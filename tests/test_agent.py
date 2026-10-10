"""Tests for the tool-calling agent loop."""
import json
from unittest.mock import Mock

import pytest

from eink_agent import agent_tools

from eink_agent.agent import (
    BLOCKED_ANSWER,
    DEGRADED_ANSWER,
    SYSTEM_PROMPT,
    run_agent,
)

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

    result = run_agent(
        "一直搜索设备",
        client=client,
        model="test-model",
        max_rounds=2,
    )

    assert result.status == "degraded"
    assert result.answer == DEGRADED_ANSWER
    assert len(result.tool_trace) == 2
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


def _response(content=None, tool_calls=None) -> Mock:
    """Build one fake chat-completion response."""
    return Mock(
        choices=[
            Mock(message=Mock(content=content, tool_calls=tool_calls or []))
        ]
    )


def _raw_tool_call(name: str, raw_arguments: str, call_id: str) -> Mock:
    """Build a tool call whose arguments are passed through unparsed."""
    function = Mock(arguments=raw_arguments)
    function.name = name

    return Mock(id=call_id, function=function)


def _tool_messages(client: Mock) -> list[dict]:
    """Return the tool messages sent in the last model request."""
    messages = client.chat.completions.create.call_args.kwargs["messages"]

    return [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]


@pytest.mark.parametrize(
    ("tool_call", "error_type"),
    [
        (_raw_tool_call("search_devices", "{not json", "c1"), "invalid_json"),
        (_raw_tool_call("search_devices", "[1, 2]", "c1"), "invalid_json"),
        (_raw_tool_call("place_order", "{}", "c1"), "unknown_tool"),
        (
            _raw_tool_call("get_device_detail", '{"device_id": -1}', "c1"),
            "invalid_arguments",
        ),
        (
            _raw_tool_call("search_devices", '{"colour": true}', "c1"),
            "invalid_arguments",
        ),
    ],
)
def test_tool_failure_is_returned_to_model_and_recorded(tool_call, error_type):
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[tool_call]),
        _response(content="请换个条件再试。"),
    ]

    result = run_agent("找设备", client=client, model="test-model")

    assert result.status == "ok"
    assert result.answer == "请换个条件再试。"

    trace = result.tool_trace[0]
    assert trace["status"] == "error"
    assert trace["error"]["type"] == error_type
    assert trace["result_count"] == 0

    payload = json.loads(_tool_messages(client)[0]["content"])
    assert payload["error"]["type"] == error_type
    assert payload["error"]["message"]


def test_invalid_arguments_error_tells_model_the_expected_arguments():
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _raw_tool_call("compare_devices", '{"device_ids": [1]}', "c1")
            ]
        ),
        _response(content="需要至少两台设备。"),
    ]

    run_agent("比较1号设备", client=client, model="test-model")

    error = json.loads(_tool_messages(client)[0]["content"])["error"]
    assert "device_ids" in error["message"]
    assert error["expected_arguments"]["required"] == ["device_ids"]


def test_unknown_tool_error_lists_available_tools():
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_raw_tool_call("place_order", "{}", "c1")]),
        _response(content="我不能下单。"),
    ]

    run_agent("帮我下单", client=client, model="test-model")

    error = json.loads(_tool_messages(client)[0]["content"])["error"]
    assert error["available_tools"] == [
        "search_devices",
        "get_device_detail",
        "compare_devices",
    ]


def test_tool_exception_is_not_leaked_to_model(monkeypatch):
    def broken_search(**_kwargs):
        raise RuntimeError("database is locked at /secret/path/eink.db")

    monkeypatch.setitem(
        agent_tools._TOOL_REGISTRY,
        "search_devices",
        (agent_tools.DeviceRequirements, broken_search),
    )
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[_raw_tool_call("search_devices", "{}", "c1")]),
        _response(content="查询暂时失败。"),
    ]

    result = run_agent("找设备", client=client, model="test-model")

    content = _tool_messages(client)[0]["content"]
    assert result.tool_trace[0]["error"]["type"] == "tool_error"
    assert "/secret/path" not in content
    assert "database is locked" not in content
    assert "/secret/path" not in json.dumps(result.tool_trace)


def test_consecutive_failures_stop_with_degraded_answer():
    client = Mock()
    bad_call = _raw_tool_call("search_devices", "{not json", "c1")
    client.chat.completions.create.side_effect = [
        _response(tool_calls=[bad_call]),
        _response(tool_calls=[bad_call]),
        _response(content="不应该走到这里"),
    ]

    result = run_agent("找设备", client=client, model="test-model")

    assert result.status == "degraded"
    assert result.answer == DEGRADED_ANSWER
    assert [trace["status"] for trace in result.tool_trace] == ["error", "error"]
    assert client.chat.completions.create.call_count == 2


def test_success_after_failure_resets_the_failure_count():
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(
            tool_calls=[
                _raw_tool_call("search_devices", '{"colour": true}', "c1")
            ]
        ),
        _response(
            tool_calls=[
                _raw_tool_call("search_devices", '{"max_price": 2000}', "c2")
            ]
        ),
        _response(content="找到两台设备。"),
    ]

    result = run_agent("两千以内", client=client, model="test-model")

    assert result.status == "ok"
    assert result.answer == "找到两台设备。"
    assert [trace["status"] for trace in result.tool_trace] == ["error", "ok"]


def test_empty_final_answer_degrades_instead_of_raising():
    client = Mock()
    client.chat.completions.create.return_value = _response(content="   ")

    result = run_agent("找设备", client=client, model="test-model")

    assert result.status == "degraded"
    assert result.answer == DEGRADED_ANSWER


def test_claimed_order_without_any_tool_call_is_rewritten():
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(content="已为您下单 1 号设备。"),
        _response(content="抱歉，我不能替您下单，可以帮您比较设备。"),
    ]

    result = run_agent("帮我买1号设备", client=client, model="test-model")

    assert result.status == "ok"
    assert result.answer == "抱歉，我不能替您下单，可以帮您比较设备。"
    assert result.tool_trace == []

    rewrite_request = client.chat.completions.create.call_args.kwargs["messages"]
    assert "已为您下单" in rewrite_request[-1]["content"]


def test_claimed_order_is_blocked_when_rewrite_still_claims_it():
    client = Mock()
    client.chat.completions.create.side_effect = [
        _response(content="已为您下单 1 号设备。"),
        _response(content="订单已提交，请留意短信。"),
    ]

    result = run_agent("帮我买1号设备", client=client, model="test-model")

    assert result.status == "blocked"
    assert result.answer == BLOCKED_ANSWER
    assert client.chat.completions.create.call_count == 2


def test_system_prompt_describes_real_data_and_unknown_fields():
    assert "fictional" not in SYSTEM_PROMPT
    assert "null" in SYSTEM_PROMPT.lower()
    assert "never infer" in SYSTEM_PROMPT


def test_tool_descriptions_no_longer_mention_fictional_devices():
    for tool in agent_tools.AGENT_TOOLS:
        assert "fictional" not in tool["function"]["description"]
