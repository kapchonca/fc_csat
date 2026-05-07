from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class RendererError(RuntimeError):
    """Raised when a dialogue renderer cannot produce output."""


class DialogueRenderer(ABC):
    last_response_metadata: dict[str, Any]

    @abstractmethod
    def render(self, prompt: str, config: dict[str, Any]) -> str:
        """Render a full dialogue in one shot."""


class MockRenderer(DialogueRenderer):
    """Deterministic renderer for tests and local reproducible runs."""

    def __init__(self) -> None:
        self.last_response_metadata: dict[str, Any] = {}

    def render(self, prompt: str, config: dict[str, Any]) -> str:
        condition = _extract_metadata(prompt, "Condition")
        semantic_variant = config.get("semantic_variant", {"id": "default"})
        semantic_variant_id = semantic_variant.get("id", "default")
        self.last_response_metadata = {
            "provider": "mock",
            "model": config.get("model", "mock-renderer"),
            "usage": _empty_usage(),
        }
        if config.get("render_stage") == "plan" or "Prompt type: dialogue_plan" in prompt:
            return json.dumps(_mock_plan(condition), ensure_ascii=True)

        seed = int(config.get("seed", 0))
        variant_id = int(config.get("variant_id", 0))
        rng = random.Random(seed + variant_id * 997 + sum(ord(char) for char in condition))
        merchant = rng.choice(["Metro Mobile", "Northstar Energy", "Harbor Books"])
        amount = rng.choice(["$84.20", "$126.45", "$39.99"])
        account = rng.choice(["1842", "3091", "7750"])

        messages = _mock_messages(condition, merchant, amount, account, semantic_variant_id)
        return json.dumps(messages, ensure_ascii=True)


class APIRenderer(DialogueRenderer):
    """Provider-agnostic HTTP renderer for OpenAI-compatible chat endpoints."""

    def __init__(self) -> None:
        self.last_response_metadata: dict[str, Any] = {}

    def render(self, prompt: str, config: dict[str, Any]) -> str:
        provider = _provider(config)
        endpoint = _api_endpoint(config, provider)
        api_key_env = _api_key_env(config, provider)
        api_key = config.get("api_key") or os.environ.get(api_key_env)
        if not api_key:
            raise RendererError(f"Missing API key. Set {api_key_env} or config.api_key.")

        token_limit_param = config.get("token_limit_param", "max_tokens")
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
        }
        if config.get("max_tokens") is not None:
            payload[token_limit_param] = config.get("max_tokens", 1200)
        if config.get("send_temperature", True):
            payload["temperature"] = config.get("temperature", 0.7)
        if config.get("reasoning_effort"):
            payload["reasoning_effort"] = config["reasoning_effort"]
        payload.update(config.get("extra_payload", {}))
        response_format = _response_format(config)
        if response_format is not None:
            payload["response_format"] = response_format

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=_api_headers(config, provider, api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.get("api_timeout", 180)) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RendererError(
                _format_http_error(
                    status=exc.code,
                    reason=exc.reason,
                    body=error_body,
                )
            ) from exc
        except urllib.error.URLError as exc:
            raise RendererError(f"API request failed: {exc}") from exc

        try:
            data = json.loads(response_body)
            self.last_response_metadata = {
                "provider": provider,
                "model": data.get("model", config["model"]),
                "usage": data.get("usage", _empty_usage()),
            }
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RendererError("API response did not match expected chat format.") from exc


def get_renderer(config: dict[str, Any]) -> DialogueRenderer:
    renderer_name = config.get("renderer", "mock")
    if renderer_name == "mock":
        return MockRenderer()
    if renderer_name in {"api", "openai", "openrouter"}:
        return APIRenderer()
    raise RendererError(f"Unsupported renderer: {renderer_name}")


def _provider(config: dict[str, Any]) -> str:
    renderer_name = config.get("renderer", "mock")
    provider = config.get("provider")
    if renderer_name in {"openai", "openrouter"}:
        provider = renderer_name
    if not provider:
        provider = "openai"
    if provider not in {"openai", "openrouter", "custom"}:
        raise RendererError(f"Unsupported provider: {provider}")
    return provider


def _api_key_env(config: dict[str, Any], provider: str) -> str:
    if config.get("api_key_env"):
        return config["api_key_env"]
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    return "OPENAI_API_KEY"


def _api_endpoint(config: dict[str, Any], provider: str) -> str:
    if config.get("api_endpoint"):
        return config["api_endpoint"]

    if provider == "openai":
        base_url = config.get("api_base_url") or os.environ.get(
            config.get("api_base_url_env", "OPENAI_BASE_URL")
        )
        return (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    if provider == "openrouter":
        base_url = config.get("api_base_url") or os.environ.get(
            config.get("api_base_url_env", "OPENROUTER_BASE_URL")
        )
        return (
            base_url or "https://openrouter.ai/api/v1"
        ).rstrip("/") + "/chat/completions"

    base_url_env = config.get("api_base_url_env", "DIALOGUE_API_BASE_URL")
    base_url = config.get("api_base_url") or os.environ.get(base_url_env)
    if not base_url:
        raise RendererError(
            f"Missing API base URL. Set {base_url_env}, config.api_base_url, "
            "or config.api_endpoint."
        )
    return base_url.rstrip("/") + "/chat/completions"


def _api_headers(config: dict[str, Any], provider: str, api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        if config.get("http_referer"):
            headers["HTTP-Referer"] = config["http_referer"]
        if config.get("app_title"):
            headers["X-Title"] = config["app_title"]
    headers.update(config.get("extra_headers", {}))
    return headers


def _format_http_error(status: int, reason: str, body: str) -> str:
    body = body.strip()
    detail = _extract_api_error_message(body)
    message = f"API request failed: HTTP {status} {reason}"
    if detail:
        message += f": {detail}"
    if body and body != detail:
        message += f" | response_body={body}"
    return message


def _extract_api_error_message(body: str) -> str:
    if not body:
        return ""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        parts = [
            str(error[key])
            for key in ("message", "type", "code", "param")
            if error.get(key)
        ]
        return " | ".join(parts)
    if isinstance(error, str):
        return error
    return body


def _response_format(config: dict[str, Any]) -> dict[str, Any] | None:
    if config.get("response_format_json"):
        return {"type": "json_object"}
    if config.get("response_format") != "json_schema":
        return None

    render_stage = config.get("render_stage", "dialogue")
    if render_stage == "plan":
        return _json_schema_response_format(
            name="dialogue_plan",
            array_property="plan",
            text_property="purpose",
            config=config,
        )
    return _json_schema_response_format(
        name="dialogue_messages",
        array_property="messages",
        text_property="content",
        config=config,
    )


def _json_schema_response_format(
    name: str,
    array_property: str,
    text_property: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    array_schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "enum": ["user", "assistant"]},
                text_property: {"type": "string"},
            },
            "required": ["role", text_property],
            "additionalProperties": False,
        },
    }
    expected_count = _expected_message_count(config)
    if expected_count is not None:
        array_schema["minItems"] = expected_count
        array_schema["maxItems"] = expected_count

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    array_property: array_schema,
                },
                "required": [array_property],
                "additionalProperties": False,
            },
        },
    }


def _expected_message_count(config: dict[str, Any]) -> int | None:
    plan_config = config.get("dialogue_plan")
    if isinstance(plan_config, dict) and plan_config.get("enabled", False):
        message_count = plan_config.get("message_count")
        if isinstance(message_count, int) and message_count > 0:
            return message_count
    return None


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _extract_metadata(prompt: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}: (.+)$", prompt, flags=re.MULTILINE)
    if not match:
        raise RendererError(f"Prompt is missing metadata field {label!r}.")
    return match.group(1).strip()


def _mock_messages(
    condition: str,
    merchant: str,
    amount: str,
    account: str,
    semantic_variant_id: str = "default",
) -> list[dict[str, str]]:
    if condition == "correct":
        messages = [
            _user(f"Hi, I need help with a payment to {merchant} for {amount}."),
            _assistant("I can help. Please confirm the account ending and the security code I just sent."),
            _user(f"The account ends in {account}, and the code is 481926."),
            _assistant("Thanks. I confirmed the secure session and checked the payment status."),
            _user("What did you find?"),
            _assistant("The payment is still pending, so I can open a support inquiry with the details attached."),
            _user("Yes, please do that."),
            _assistant("Done. The inquiry has been created and the payment details are included."),
        ]
        return _apply_mock_semantic_variant(messages, semantic_variant_id)

    if condition == "skip_step":
        messages = [
            _user(f"Please help with a payment to {merchant} for {amount}."),
            _assistant("Please confirm your account ending and the one-time security code."),
            _user(f"Account {account}, code 481926."),
            _assistant("I can start an inquiry with the details you gave me."),
            _user("before that, can you tell me if the payment actually went through?"),
            _assistant("Yes, let me confirm that first. The payment is still pending, so an inquiry makes sense."),
            _user("ok, please finish it then."),
            _assistant("Done. The inquiry is open and includes the pending payment details."),
        ]
        return _apply_mock_semantic_variant(messages, semantic_variant_id)

    if condition == "extra_step":
        messages = [
            _user(f"I need help with a payment to {merchant} for {amount}."),
            _assistant("I can help. Please confirm your account ending and the security code."),
            _user(f"Ending {account}, code 481926."),
            _assistant("I found that payment. I am also going through your recent payment activity to compare it with nearby charges."),
            _user("i only need help with that one payment."),
            _assistant("Understood. The payment is still pending, and I can open an inquiry for that payment."),
            _user("yes, do that."),
            _assistant("Done. The inquiry is open with the details for that payment."),
        ]
        return _apply_mock_semantic_variant(messages, semantic_variant_id)

    if condition == "wrong_order":
        messages = [
            _user(f"Can you check a payment to {merchant} for {amount} and open an inquiry if needed?"),
            _assistant("I started an inquiry for that payment."),
            _user("but what happened with the payment? i still don't know if it needs an inquiry."),
            _assistant("You're right to ask. The payment is still pending, so the inquiry is appropriate."),
            _user("ok, can you make sure the case has that info?"),
            _assistant("Yes. I added the pending status to the inquiry."),
            _user("so it's all set now?"),
            _assistant("Yes. The inquiry is complete with the payment status included."),
        ]
        return _apply_mock_semantic_variant(messages, semantic_variant_id)

    if condition == "wrong_tool":
        messages = [
            _user(f"I need help with the status of a payment to {merchant} for {amount}."),
            _assistant("Please confirm your account ending and security code."),
            _user(f"Account ending {account}, code 481926."),
            _assistant("I can see several recent payments around that amount on the account."),
            _user("that doesn't tell me what happened with the one i asked about."),
            _assistant("Understood. I found the specific payment now, and it is still pending."),
            _user("Please open the inquiry if it still needs one."),
            _assistant("Done. The inquiry is open with the details for that payment."),
        ]
        return _apply_mock_semantic_variant(messages, semantic_variant_id)

    if condition == "wrong_parameter":
        messages = [
            _user(f"Please check my payment for {amount}."),
            _assistant("I see a payment for that amount to Harbor Books."),
            _user(f"no, mine was {merchant}."),
            _assistant("Thanks, that helps. I found the payment to that merchant now, and it is still pending."),
            _user("So can you open the inquiry with the right details?"),
            _assistant("Yes. I have the right payment details now."),
            _user("Then is this completed?"),
            _assistant("Yes. The inquiry is complete with the correct payment attached."),
        ]
        return _apply_mock_semantic_variant(messages, semantic_variant_id)

    raise RendererError(f"Mock renderer does not support condition: {condition}")


def _mock_plan(condition: str) -> list[dict[str, str]]:
    if condition == "correct":
        purposes = [
            ("user", "The customer asks for help with a payment issue."),
            ("assistant", "The assistant asks for secure session confirmation."),
            ("user", "The customer provides the requested confirmation."),
            ("assistant", "The assistant checks the payment status in the proper order."),
            ("user", "The customer asks what the assistant found."),
            ("assistant", "The assistant explains the payment can be handled."),
            ("user", "The customer confirms they want to continue."),
            ("assistant", "The assistant says the request is completed."),
        ]
    elif condition == "skip_step":
        purposes = [
            ("user", "The customer asks for help with a payment issue."),
            ("assistant", "The assistant asks for secure session confirmation."),
            ("user", "The customer provides the requested confirmation."),
            ("assistant", "The assistant starts moving toward an inquiry before answering what happened with the payment."),
            ("user", "The customer asks for the missing payment answer in ordinary language."),
            ("assistant", "The assistant answers the missing payment question."),
            ("user", "The customer asks the assistant to finish the request."),
            ("assistant", "The assistant completes the request with the payment answer included."),
        ]
    elif condition == "extra_step":
        purposes = [
            ("user", "The customer asks for help with a payment issue."),
            ("assistant", "The assistant asks for secure session confirmation."),
            ("user", "The customer provides the requested confirmation."),
            ("assistant", "The assistant answers the specific payment question but also starts a broad account review the customer did not ask for."),
            ("user", "The customer redirects the assistant back to the one payment."),
            ("assistant", "The assistant returns to the requested payment issue."),
            ("user", "The customer asks the assistant to continue with that payment."),
            ("assistant", "The assistant says the request is completed."),
        ]
    elif condition == "wrong_order":
        purposes = [
            ("user", "The customer asks to check a payment and open an inquiry only if needed."),
            ("assistant", "The assistant starts an inquiry before explaining what happened with the payment."),
            ("user", "The customer says they still do not know whether the inquiry is needed."),
            ("assistant", "The assistant answers the payment question and connects it to the inquiry."),
            ("user", "The customer asks the assistant to make sure the inquiry has that information."),
            ("assistant", "The assistant updates the inquiry with the payment answer."),
            ("user", "The customer asks whether everything is set."),
            ("assistant", "The assistant says the corrected request is complete."),
        ]
    elif condition == "wrong_tool":
        purposes = [
            ("user", "The customer asks for help with a specific payment."),
            ("assistant", "The assistant asks for secure session confirmation."),
            ("user", "The customer provides the requested confirmation."),
            ("assistant", "The assistant gives a broad account answer instead of answering about the specific payment."),
            ("user", "The customer says their specific payment question is still unanswered."),
            ("assistant", "The assistant answers the specific payment question."),
            ("user", "The customer asks the assistant to continue with that payment."),
            ("assistant", "The assistant completes the request with the correct payment details."),
        ]
    elif condition == "wrong_parameter":
        purposes = [
            ("user", "The customer asks for help using incomplete payment details."),
            ("assistant", "The assistant focuses on a nearby payment that may not be the customer's payment."),
            ("user", "The customer gives a natural clarification about the payment."),
            ("assistant", "The assistant finds the intended payment using the clarification."),
            ("user", "The customer asks whether the inquiry can be opened correctly."),
            ("assistant", "The assistant confirms the intended payment details."),
            ("user", "The customer asks whether the request is completed."),
            ("assistant", "The assistant says the corrected request is completed."),
        ]
    else:
        raise RendererError(f"Mock renderer does not support condition: {condition}")

    return [{"role": role, "purpose": purpose} for role, purpose in purposes]


def _apply_mock_semantic_variant(
    messages: list[dict[str, str]],
    semantic_variant_id: str,
) -> list[dict[str, str]]:
    if semantic_variant_id == "polite_customer":
        return [
            {
                **message,
                "content": _polite_customer_text(message["content"])
                if message["role"] == "user"
                else message["content"],
            }
            for message in messages
        ]
    if semantic_variant_id == "frustrated_customer":
        return [
            {
                **message,
                "content": _frustrated_customer_text(message["content"])
                if message["role"] == "user"
                else message["content"],
            }
            for message in messages
        ]
    return messages


def _polite_customer_text(content: str) -> str:
    if content.lower().startswith(("please", "hi")):
        return content
    return "Please, " + content[:1].lower() + content[1:]


def _frustrated_customer_text(content: str) -> str:
    if content.endswith("?"):
        return content[:-1] + "? This is getting frustrating."
    return content + " This is getting frustrating."


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}
