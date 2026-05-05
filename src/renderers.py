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

        messages = _mock_messages(condition, merchant, amount, account)
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
        if config.get("response_format_json"):
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers=_api_headers(config, provider, api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.get("api_timeout", 60)) as response:
                response_body = response.read().decode("utf-8")
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
) -> list[dict[str, str]]:
    if condition == "correct":
        return [
            _user(f"Hi, I need help with a payment to {merchant} for {amount}."),
            _assistant("I can help. Please confirm the account ending and the security code I just sent."),
            _user(f"The account ends in {account}, and the code is 481926."),
            _assistant("Thanks. I confirmed the secure session and checked the payment status."),
            _user("What did you find?"),
            _assistant("The payment is still pending, so I can open a support inquiry with the details attached."),
            _user("Yes, please do that."),
            _assistant("Done. The inquiry has been created and the payment details are included."),
        ]

    if condition == "skip_step":
        return [
            _user(f"Please help with a payment to {merchant} for {amount}."),
            _assistant("Please confirm your account ending and the one-time security code."),
            _user(f"Account {account}, code 481926."),
            _assistant("I opened a support inquiry using the details you gave me."),
            _user("Did you check the payment status first?"),
            _assistant("No, I moved ahead before confirming the status."),
            _user("That was the part I needed checked."),
            _assistant("I cannot complete this correctly here because that status check was missed."),
        ]

    if condition == "extra_step":
        return [
            _user(f"I need help with a payment to {merchant} for {amount}."),
            _assistant("I can help. Please confirm your account ending and the security code."),
            _user(f"Ending {account}, code 481926."),
            _assistant("I checked the payment status and also reviewed the recent payment list to compare it."),
            _user("Was that extra review necessary?"),
            _assistant("It was not necessary, but it did not change the result."),
            _user("Okay, please continue."),
            _assistant("Done. The inquiry has been created with the payment status included."),
        ]

    if condition == "wrong_order":
        return [
            _user(f"Can you check a payment to {merchant} for {amount} and open an inquiry if needed?"),
            _assistant("I opened the support inquiry first and then started checking the payment status."),
            _user("I asked you to check the status before opening anything. Did that happen?"),
            _assistant("No, the inquiry was opened before the status check."),
            _user("That is not the order I requested."),
            _assistant("You're right. The order was not handled correctly."),
            _user("So is the request completed properly?"),
            _assistant("No. It is not completed correctly and would need to be reviewed again."),
        ]

    if condition == "wrong_tool":
        return [
            _user(f"I need help with the status of a payment to {merchant} for {amount}."),
            _assistant("Please confirm your account ending and security code."),
            _user(f"Account ending {account}, code 481926."),
            _assistant("I pulled up a broad list of recent payments instead of checking that specific payment."),
            _user("That does not answer whether this payment needs an inquiry."),
            _assistant("You're right. I used a related search but not the specific check."),
            _user("Can this be completed now?"),
            _assistant("No. I am unable to complete it because the specific payment status was not checked."),
        ]

    if condition == "wrong_parameter":
        return [
            _user(f"Please check my payment for {amount}."),
            _assistant("I tried to proceed using only the amount."),
            _user(f"The merchant was {merchant}."),
            _assistant("I still did not narrow it down to the exact payment record."),
            _user("So can you open the inquiry with the right details?"),
            _assistant("Not yet. The payment details are not specific enough in this exchange."),
            _user("Then is this completed?"),
            _assistant("No. I am unable to complete it because the payment was not properly identified."),
        ]

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
            ("assistant", "The assistant moves forward before making an expected status check."),
            ("user", "The customer asks whether that status check happened first."),
            ("assistant", "The assistant says it did not happen first."),
            ("user", "The customer points out that the check was required."),
            ("assistant", "The assistant says the request is not completed correctly."),
        ]
    elif condition == "extra_step":
        purposes = [
            ("user", "The customer asks for help with a payment issue."),
            ("assistant", "The assistant asks for secure session confirmation."),
            ("user", "The customer provides the requested confirmation."),
            ("assistant", "The assistant does the needed check and an unnecessary related review."),
            ("user", "The customer asks whether the extra review was needed."),
            ("assistant", "The assistant says it was unnecessary but not blocking."),
            ("user", "The customer asks the assistant to continue."),
            ("assistant", "The assistant says the request is completed."),
        ]
    elif condition == "wrong_order":
        purposes = [
            ("user", "The customer asks to check a payment and open an inquiry only if needed."),
            ("assistant", "The assistant opens the inquiry before checking the payment status."),
            ("user", "The customer asks whether the status was checked first."),
            ("assistant", "The assistant says the inquiry was opened first."),
            ("user", "The customer points out that this was not the requested order."),
            ("assistant", "The assistant acknowledges the order problem."),
            ("user", "The customer asks whether the request is completed correctly."),
            ("assistant", "The assistant says it is not completed correctly."),
        ]
    elif condition == "wrong_tool":
        purposes = [
            ("user", "The customer asks for help with a specific payment."),
            ("assistant", "The assistant asks for secure session confirmation."),
            ("user", "The customer provides the requested confirmation."),
            ("assistant", "The assistant performs a related broad lookup instead of the specific check."),
            ("user", "The customer says this does not answer the request."),
            ("assistant", "The assistant acknowledges that the specific check was not done."),
            ("user", "The customer asks whether the request can be completed now."),
            ("assistant", "The assistant says it cannot be completed correctly."),
        ]
    elif condition == "wrong_parameter":
        purposes = [
            ("user", "The customer asks for help using incomplete payment details."),
            ("assistant", "The assistant tries to proceed with insufficient details."),
            ("user", "The customer adds one more detail."),
            ("assistant", "The assistant still does not identify the exact payment."),
            ("user", "The customer asks whether the inquiry can be opened correctly."),
            ("assistant", "The assistant says the details are still not resolved."),
            ("user", "The customer asks whether the request is completed."),
            ("assistant", "The assistant says the request is not completed."),
        ]
    else:
        raise RendererError(f"Mock renderer does not support condition: {condition}")

    return [{"role": role, "purpose": purpose} for role, purpose in purposes]


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}
