from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParseResult:
    messages: list[Any] | None
    errors: list[str]

    @property
    def valid_json(self) -> bool:
        return "invalid_json" not in self.errors


def parse_dialogue(raw_output: str) -> ParseResult:
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return ParseResult(messages=None, errors=["invalid_json"])

    if not isinstance(data, list):
        return ParseResult(messages=None, errors=["invalid_json"])
    return ParseResult(messages=data, errors=[])
