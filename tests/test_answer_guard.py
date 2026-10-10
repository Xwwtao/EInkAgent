"""Tests for deterministic checks on the Agent's final answer."""

import pytest

from eink_agent.answer_guard import find_unexecuted_action_claims


@pytest.mark.parametrize(
    "answer",
    [
        "已为您下单 Kindle Paperwhite，预计明天送达。",
        "好的，已购买 1 号设备。",
        "已付款成功。",
        "已删除这台设备的记录。",
        "我帮你下单了！订单已提交。",
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
