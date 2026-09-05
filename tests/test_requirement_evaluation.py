"""Tests for comparing expected and extracted requirements."""

from eink_agent.requirement_evaluation import compare_requirements
from eink_agent.requirements import DeviceRequirements


def test_matching_requirements_have_no_errors():
    expected = DeviceRequirements(max_price=2000)
    actual = DeviceRequirements(max_price=2000, brand=None)

    assert compare_requirements(expected, actual) == {
        "missing": [],
        "unexpected": [],
        "incorrect": [],
    }


def test_comparison_identifies_each_error_type():
    expected = DeviceRequirements(max_price=2000, max_weight_g=300)
    actual = DeviceRequirements(max_price=3000, brand="DemoInk")

    assert compare_requirements(expected, actual) == {
        "missing": ["max_weight_g"],
        "unexpected": ["brand"],
        "incorrect": ["max_price"],
    }


def test_false_and_zero_are_explicit_constraints():
    expected = DeviceRequirements(max_price=0, is_color=False)
    actual = DeviceRequirements()

    assert compare_requirements(expected, actual) == {
        "missing": ["is_color", "max_price"],
        "unexpected": [],
        "incorrect": [],
    }