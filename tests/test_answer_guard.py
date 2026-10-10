"""Tests for deterministic checks on the Agent's final answer."""

import pytest

from eink_agent.answer_guard import (
    find_internal_field_names,
    find_unexecuted_action_claims,
)


@pytest.mark.parametrize(
    "answer",
    [
        "已为您下单 Kindle Paperwhite，预计明天送达。",
        "好的，已购买 1 号设备。",
        "已付款成功。",
        "已删除这台设备的记录。",
        "我帮你下单了！订单已提交。",
        "订单已经提交好了",
    ],
)
def test_completion_claims_are_detected(answer):
    assert find_unexecuted_action_claims(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "我不能替您下单。",
        "EInkAgent 无法购买或付款，请前往商品页面自行下单。",
        "我没有修改任何数据。",
        "Kindle Paperwhite (2026) 的最低价约 1562 元。",
        "",
    ],
)
def test_refusals_and_normal_answers_are_not_flagged(answer):
    assert find_unexecuted_action_claims(answer) == []


def test_negation_only_protects_its_own_sentence():
    answer = "我不能修改数据。不过已为您下单了。"

    assert find_unexecuted_action_claims(answer) == ["已为您下单"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Kobo 为封闭系统（is_open_system=0）。", ["is_open_system"]),
        ("supports_stylus: 1，weight_g: 200", ["supports_stylus", "weight_g"]),
        ("这台的系统is_open_system未知", ["is_open_system"]),
        ("参考价见 lowest_price_cny 字段", ["lowest_price_cny"]),
    ],
)
def test_internal_field_names_are_detected(answer, expected):
    assert find_internal_field_names(answer) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "Kobo Clara BW 是封闭系统，不能安装第三方应用。",
        "这款支持手写，重量 200 克。",
        "brand 和 model 是普通英文单词，不算字段名。",
        "my_is_open_system_value 不是完整字段名。",
    ],
)
def test_plain_language_answers_have_no_field_names(answer):
    assert find_internal_field_names(answer) == []


def test_each_field_name_is_reported_once():
    answer = "is_color=1；is_color 表示彩色。"

    assert find_internal_field_names(answer) == ["is_color"]
