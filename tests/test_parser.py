from __future__ import annotations

from src.parser import parse_dialogue, parse_dialogue_plan


def test_parse_dialogue_accepts_structured_output_wrapper() -> None:
    result = parse_dialogue(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}'
    )

    assert result.errors == []
    assert result.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_parse_dialogue_plan_accepts_structured_output_wrapper() -> None:
    result = parse_dialogue_plan(
        '{"plan":[{"role":"user","purpose":"ask"},{"role":"assistant","purpose":"answer"}]}'
    )

    assert result.errors == []
    assert result.items == [
        {"role": "user", "purpose": "ask"},
        {"role": "assistant", "purpose": "answer"},
    ]
