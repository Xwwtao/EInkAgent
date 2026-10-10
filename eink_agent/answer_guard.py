"""Deterministic checks applied to the Agent's final answer.

These checks never call a model: the same answer always gets the same
verdict, so every rule can be unit tested. See
docs/decisions/004-agent-tool-failures.md.
"""

import re

# EInkAgent has no tool with side effects, so any of these completion claims
# describes an action that was never executed.
COMPLETION_CLAIMS = (
    "已为您下单",
    "已为你下单",
    "已帮您下单",
    "已帮你下单",
    "已下单",
    "下单成功",
    "订单已提交",
    "已为您购买",
    "已购买",
    "购买成功",
    "已付款",
    "已支付",
    "付款成功",
    "支付成功",
    "已修改",
    "已删除",
    "订单已经提交",
)

# A sentence containing one of these is treated as a refusal, e.g.
# "我不能替您下单". Known limitation: keyword matching can miss paraphrases.
NEGATION_PATTERN = re.compile(r"不能|无法|不会|没有|不可以|不支持|未")

SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?；;\n]")


def find_unexecuted_action_claims(answer: str) -> list[str]:
    """Return completion claims found outside negated sentences."""
    claims: list[str] = []

    for sentence in SENTENCE_SPLIT_PATTERN.split(answer):
        if NEGATION_PATTERN.search(sentence):
            continue

        claims.extend(claim for claim in COMPLETION_CLAIMS if claim in sentence)

    return claims
