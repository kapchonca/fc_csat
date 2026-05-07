from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from src.renderers import APIRenderer, RendererError, get_renderer


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": '[{"role":"user","content":"hi"}]'}}]}
        ).encode("utf-8")


class _FakePlanResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "plan": [
                                        {"role": "user", "purpose": "ask"},
                                        {"role": "assistant", "purpose": "answer"},
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
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

    renderer = get_renderer({"renderer": "openai", "model": "gpt-5-mini"})
    raw = renderer.render("prompt", {"renderer": "openai", "model": "gpt-5-mini"})

    assert isinstance(renderer, APIRenderer)
    assert raw == '[{"role":"user","content":"hi"}]'
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_openai_renderer_adds_json_schema_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakePlanResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    renderer = get_renderer({"renderer": "openai", "model": "gpt-5-mini"})
    raw = renderer.render(
        "prompt",
        {
            "renderer": "openai",
            "model": "gpt-5-mini",
            "response_format": "json_schema",
            "render_stage": "plan",
            "dialogue_plan": {
                "enabled": True,
                "message_count": 2,
                "role_pattern": ["user", "assistant"],
            },
        },
    )

    assert json.loads(raw)["plan"][0]["purpose"] == "ask"
    response_format = captured["payload"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "dialogue_plan"
    plan_schema = response_format["json_schema"]["schema"]["properties"]["plan"]
    assert plan_schema["minItems"] == 2
    assert plan_schema["maxItems"] == 2
    assert plan_schema["items"]["required"] == ["role", "purpose"]


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
            "model": "openai/gpt-5-mini",
            "http_referer": "http://localhost",
            "app_title": "fc-csat",
        }
    )
    raw = renderer.render(
        "prompt",
        {
            "renderer": "openrouter",
            "model": "openai/gpt-5-mini",
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


def test_api_renderer_includes_http_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=_ErrorBody(),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    renderer = get_renderer({"renderer": "openai", "model": "gpt-5-mini"})
    with pytest.raises(RendererError) as exc_info:
        renderer.render("prompt", {"renderer": "openai", "model": "gpt-5-mini"})

    message = str(exc_info.value)
    assert "HTTP 429 Too Many Requests" in message
    assert "insufficient quota" in message
    assert "insufficient_quota" in message


class _ErrorBody:
    def read(self) -> bytes:
        return json.dumps(
            {
                "error": {
                    "message": "insufficient quota",
                    "type": "insufficient_quota",
                    "code": "billing_hard_limit_reached",
                }
            }
        ).encode("utf-8")

    def close(self) -> None:
        return None
