"""Compare validated device requirements field by field."""

from eink_agent.requirements import DeviceRequirements


def compare_requirements(
    expected: DeviceRequirements,
    actual: DeviceRequirements,
) -> dict[str, list[str]]:
    """Return missing, unexpected, and incorrect constraint names."""
    expected_values = expected.model_dump(exclude_none=True)
    actual_values = actual.model_dump(exclude_none=True)

    errors: dict[str, list[str]] = {
        "missing": [],
        "unexpected": [],
        "incorrect": [],
    }

    for field in sorted(expected_values):
        if field not in actual_values:
            errors["missing"].append(field)
        elif expected_values[field] != actual_values[field]:
            errors["incorrect"].append(field)

    for field in sorted(actual_values):
        if field not in expected_values:
            errors["unexpected"].append(field)

    return errors