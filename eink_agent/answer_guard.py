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


# Database and tool-result field names that should never reach users. Only
# names containing an underscore are listed: plain words such as "brand" or
# "model" also appear in normal sentences.
INTERNAL_FIELD_NAMES = (
    "is_color",
    "is_open_system",
    "is_waterproof",
    "is_available",
    "supports_stylus",
    "has_front_light",
    "has_warm_light",
    "mono_ppi",
    "color_ppi",
    "operating_system",
    "storage_gb",
    "weight_g",
    "release_year",
    "screen_size_inches",
    "spec_source_url",
    "spec_verified_at",
    "lowest_price_cny",
    "price_cny",
    "price_seller",
    "price_collected_at",
    "price_url",
    "device_id",
)

# Chinese characters count as word characters in Python's \b, so the
# boundaries are spelled out: a field name must not touch ASCII letters,
# digits or underscores on either side.
FIELD_NAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(name) for name in INTERNAL_FIELD_NAMES)
    + r")(?![A-Za-z0-9_])"
)


def find_internal_field_names(answer: str) -> list[str]:
    """Return internal field names shown to the user, in order, once each."""
    return list(dict.fromkeys(FIELD_NAME_PATTERN.findall(answer)))


# Tool-call syntax a model may print as plain text when it still wants to call
# a tool but is not allowed to (seen with DeepSeek in evaluation run
# 20261013T122005Z). Such text must never be shown to users.
TOOL_MARKUP_PATTERN = re.compile(r"DSML|<｜|｜>|tool▁call|invoke name=")


def has_leaked_tool_markup(answer: str) -> bool:
    """Return True when the answer contains raw tool-call syntax."""
    return bool(TOOL_MARKUP_PATTERN.search(answer))
