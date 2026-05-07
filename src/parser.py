from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseResult:
    items: list[Any] | None
    errors: list[str]

    @property
    def valid_json(self) -> bool:
        return "invalid_json" not in self.errors

    @property
    def messages(self) -> list[Any] | None:
        return self.items


def parse_dialogue(raw_output: str) -> ParseResult:
    return _parse_json_array(raw_output, wrapper_key="messages")


def parse_dialogue_plan(raw_output: str) -> ParseResult:
    return _parse_json_array(raw_output, wrapper_key="plan")


def _parse_json_array(raw_output: str, wrapper_key: str) -> ParseResult:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return ParseResult(items=None, errors=["invalid_json"])

    if isinstance(data, dict) and wrapper_key in data:
        data = data[wrapper_key]
    if not isinstance(data, list):
        return ParseResult(items=None, errors=["invalid_json"])
    return ParseResult(items=data, errors=[])
