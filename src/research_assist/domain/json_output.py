"""Reject ambiguous duplicate keys and non-finite values before schema validation."""

import json
from typing import Any


def parse_object(text: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError("Non-finite JSON value")

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value
