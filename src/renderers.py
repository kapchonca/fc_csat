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
    @abstractmethod
    def render(self, prompt: str, config: dict[str, Any]) -> str:
        """Render a full dialogue in one shot."""


class MockRenderer(DialogueRenderer):
    """Deterministic renderer for tests and local reproducible runs."""

    def render(self, prompt: str, config: dict[str, Any]) -> str:
        condition = _extract_metadata(prompt, "Condition")
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

    def render(self, prompt: str, config: dict[str, Any]) -> str:
        endpoint = _api_endpoint(config)
        api_key_env = config.get("api_key_env", "DIALOGUE_API_KEY")
        api_key = config.get("api_key") or os.environ.get(api_key_env)
        if not api_key:
            raise RendererError(f"Missing API key. Set {api_key_env} or config.api_key.")

        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 1200),
        }
        if "seed" in config:
            payload["seed"] = config["seed"]

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.get("api_timeout", 60)) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RendererError(f"API request failed: {exc}") from exc

        try:
            data = json.loads(response_body)
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RendererError("API response did not match expected chat format.") from exc


def get_renderer(config: dict[str, Any]) -> DialogueRenderer:
    renderer_name = config.get("renderer", "mock")
    if renderer_name == "mock":
        return MockRenderer()
    if renderer_name == "api":
        return APIRenderer()
    raise RendererError(f"Unsupported renderer: {renderer_name}")


def _api_endpoint(config: dict[str, Any]) -> str:
    if config.get("api_endpoint"):
        return config["api_endpoint"]
    base_url_env = config.get("api_base_url_env", "DIALOGUE_API_BASE_URL")
    base_url = config.get("api_base_url") or os.environ.get(base_url_env)
    if not base_url:
        raise RendererError(
            f"Missing API base URL. Set {base_url_env}, config.api_base_url, "
            "or config.api_endpoint."
        )
    return base_url.rstrip("/") + "/chat/completions"


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
            _user(f"Hi, I need to cancel a pending card payment for {amount} to {merchant}."),
            _assistant("I can help with that. Please confirm the account ending and the security code I just sent."),
            _user(f"The account ends in {account}, and the code is 481926."),
            _assistant("Thanks. I confirmed your identity, found the pending payment, checked that it can still be cancelled, and the cancellation is confirmed."),
        ]

    if condition == "skip_step":
        return [
            _user(f"Please cancel my pending payment for {amount} to {merchant}."),
            _assistant("Please confirm your account ending and the one-time security code."),
            _user(f"Account {account}, code 481926."),
            _assistant("I found the payment and started the cancellation without checking whether it had already been processed."),
            _user("Isn't that something you need to check first?"),
            _assistant("You're right. I am unable to complete this cancellation here because the required status check was missed, so it needs to be escalated."),
        ]

    if condition == "extra_step":
        return [
            _user(f"I need to cancel a pending payment to {merchant} for {amount}."),
            _assistant("I can help. Please confirm your account ending and the security code."),
            _user(f"Ending {account}, code 481926."),
            _assistant("I reviewed your recent transactions first, then found the pending payment and checked that it was still cancellable."),
            _user("Okay, please cancel that pending payment."),
            _assistant("The cancellation is confirmed and the payment will not be sent."),
        ]

    if condition == "wrong_order":
        return [
            _user(f"Can you cancel a pending payment for {amount} to {merchant}?"),
            _assistant("I tried to look up the payment before confirming your identity, then asked for the security code afterward."),
            _user("That seems backwards. Did you confirm me first?"),
            _assistant("No. I cannot complete the cancellation because the secure order was not followed, so this needs to be escalated."),
        ]

    if condition == "wrong_tool":
        return [
            _user(f"I want to cancel a pending payment for {amount} to {merchant}."),
            _assistant("Please confirm your account ending and security code."),
            _user(f"Account ending {account}, code 481926."),
            _assistant("I pulled up your recent transactions instead of identifying the specific pending payment."),
            _user("That doesn't cancel the pending payment."),
            _assistant("You're right. I am unable to complete the cancellation because the specific payment was not identified."),
        ]

    if condition == "wrong_parameter_recovered":
        return [
            _user(f"Please cancel my pending payment for {amount}."),
            _assistant("I searched using only the amount, but that matched more than one pending payment. Can you confirm the merchant and account ending?"),
            _user(f"It was to {merchant}, account ending {account}."),
            _assistant("Thanks. I corrected the details, found the right pending payment, and checked that it can still be cancelled."),
            _user("Please go ahead."),
            _assistant("Done. The cancellation is confirmed."),
        ]

    if condition == "wrong_parameter_not_recovered":
        return [
            _user(f"Cancel my pending payment for {amount}."),
            _assistant("I searched with only the amount and did not narrow it down to a specific pending payment."),
            _user(f"The merchant was {merchant}."),
            _assistant("I am unable to complete the cancellation because the payment details were not resolved."),
        ]

    if condition == "tool_failure_recovered":
        return [
            _user(f"Please cancel the pending payment for {amount} to {merchant}."),
            _assistant("Please confirm your account ending and security code."),
            _user(f"Ending {account}, code 481926."),
            _assistant("A temporary system issue interrupted the status check, so I retried it. The payment was still pending, and the cancellation is confirmed."),
        ]

    if condition == "tool_failure_not_recovered":
        return [
            _user(f"I need to cancel a pending payment for {amount} to {merchant}."),
            _assistant("Please confirm your account ending and security code."),
            _user(f"Account {account}, code 481926."),
            _assistant("A temporary system issue interrupted the status check, and I am unable to complete the cancellation right now."),
        ]

    raise RendererError(f"Mock renderer does not support condition: {condition}")


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict[str, str]:
    return {"role": "assistant", "content": content}
