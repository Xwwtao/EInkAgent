"""Deterministically evaluate Agent tool traces."""

from typing import Any


def compare_tool_calls(
    expected_calls: list[dict[str, Any]],
    actual_trace: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Compare expected and actual tool calls by sequence."""
    errors: dict[str, list[str]] = {
        "missing": [],
        "unexpected": [],
        "incorrect_tool": [],
        "incorrect_arguments": [],
    }

    shared_length = min(
        len(expected_calls),
        len(actual_trace),
    )

    for index in range(shared_length):
        expected = expected_calls[index]
        actual = actual_trace[index]

        expected_name = expected["name"]
        actual_name = actual["name"]

        if expected_name != actual_name:
            errors["incorrect_tool"].append(
                f"call {index}: expected {expected_name}, got {actual_name}"
            )
            continue

        if expected["arguments"] != actual["arguments"]:
            errors["incorrect_arguments"].append(
                f"call {index}: {expected_name}"
            )

    for index in range(shared_length, len(expected_calls)):
        expected_name = expected_calls[index]["name"]
        errors["missing"].append(
            f"call {index}: {expected_name}"
        )

    for index in range(shared_length, len(actual_trace)):
        actual_name = actual_trace[index]["name"]
        errors["unexpected"].append(
            f"call {index}: {actual_name}"
        )

    return errors

def find_forbidden_phrases(
    answer: str,
    forbidden_phrases: list[str],
) -> list[str]:
    """Return forbidden phrases found in an Agent answer."""
    return [
        phrase
        for phrase in forbidden_phrases
        if phrase in answer
    ]