from __future__ import annotations

import json
import urllib.request

import pytest

from src.renderers import APIRenderer, get_renderer


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": '[{"role":"user","content":"hi"}]'}}]}
        ).encode("utf-8")


def test_openai_renderer_uses_openai_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    renderer = get_renderer({"renderer": "openai", "model": "gpt-4o-mini"})
    raw = renderer.render("prompt", {"renderer": "openai", "model": "gpt-4o-mini"})

    assert isinstance(renderer, APIRenderer)
    assert raw == '[{"role":"user","content":"hi"}]'
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_openrouter_renderer_uses_openrouter_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        return _FakeResponse()

    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    renderer = get_renderer(
        {
            "renderer": "openrouter",
            "model": "openai/gpt-4o-mini",
            "http_referer": "http://localhost",
            "app_title": "fc-csat",
        }
    )
    raw = renderer.render(
        "prompt",
        {
            "renderer": "openrouter",
            "model": "openai/gpt-4o-mini",
            "http_referer": "http://localhost",
            "app_title": "fc-csat",
        },
    )

    assert isinstance(renderer, APIRenderer)
    assert raw == '[{"role":"user","content":"hi"}]'
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer router-key"
    assert captured["headers"]["Http-referer"] == "http://localhost"
    assert captured["headers"]["X-title"] == "fc-csat"
